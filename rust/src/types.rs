//! What the walk learns about a type from its MRO, cached per type.
//!
//! The Python walk rereads a value's MRO through `type.__dict__`
//! descriptors for every value. The walker reads it once per type and
//! call, and keeps the type alive so its address is not reused. Native code
//! runs no application code; a class that code run inside a call changes
//! (a signal handler, a collection, another thread) is seen from the next
//! call, an accepted difference (`rust/DESIGN.md`).

use std::sync::Arc;

use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::gc::PyVisit;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple, PyType};
use pyo3::PyTraverseError;

use crate::statics::{address, type_address, Statics};

/// A builtin container a value keeps its entries in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Base {
    Dict,
    List,
    Tuple,
    Set,
    FrozenSet,
    /// Python 3.15's `frozendict`.
    FrozenDict,
}

/// A builtin scalar a subclass instance keeps its value in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Scalar {
    Str,
    Bytes,
    Int,
    Float,
}

/// How the walk copies instances of a type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    /// A str, bytes, int, or float subclass (`_scalar_copy_type`).
    Scalar(Scalar),
    /// A builtin container or a subclass of one (`_container_base`).
    Container(Base),
    /// Anything else becomes a `FakeClass`.
    Object,
}

/// Member descriptors and named tuple fields (`ValuesMixin._layout`).
pub(crate) struct Layout {
    /// Each member's name and bound `member.__get__`.
    pub members: Vec<(Py<PyAny>, Py<PyAny>)>,
    pub fields: Vec<Py<PyAny>>,
}

/// The cached facts about one type.
pub(crate) struct TypeInfo {
    /// Keeps the type, and with it its address, alive.
    pub ty: Py<PyType>,
    pub kind: Kind,
    /// Whether `OrderedDict` is in the MRO (`_derives`).
    pub ordered: bool,
    /// Bound `__get__` of the real instance-dict descriptor, if the first
    /// `__dict__` entry along the MRO is one (`_attributes`).
    pub dict_get: Option<Py<PyAny>>,
    pub layout: Option<Arc<Layout>>,
    /// Whether a class dict along the MRO has a key that is not an exact
    /// str, whose own `__eq__` or `__hash__` a lookup or set may run, or
    /// a getset other than the instance dict's own planted as `__dict__`,
    /// whose getter may call application code. The Python walk must.
    pub hostile: bool,
}

impl TypeInfo {
    /// The kind of an exact builtin container type, without a cache lookup.
    pub(crate) fn exact_kind(s: &Statics, type_address: usize) -> Option<Kind> {
        let base = match type_address {
            a if a == s.list => Base::List,
            a if a == s.dict => Base::Dict,
            a if a == s.tuple => Base::Tuple,
            a if a == s.set => Base::Set,
            a if a == s.frozenset => Base::FrozenSet,
            _ => return None,
        };
        Some(Kind::Container(base))
    }

    /// Reads a type's facts from its MRO, as the Python helpers do.
    pub(crate) fn compute(py: Python<'_>, s: &Statics, ty: &Bound<'_, PyType>) -> PyResult<Self> {
        let mro = s.class_mro.bind(py).call1((ty, ty.get_type()))?;
        let mut scalar = None;
        let mut base = None;
        let mut ordered = false;
        let mut hostile = false;
        let mut namespaces = Vec::new();
        for cls in mro.try_iter()? {
            let cls = cls?;
            let cls_address = address(&cls);
            scalar = scalar.or_else(|| s.scalar_of(cls_address));
            base = base.or_else(|| s.base_of(cls_address));
            ordered |= cls_address == s.ordered_dict;
            // Iterating a class dict yields its keys and runs nothing else.
            let namespace = s.class_dict.bind(py).call1((&cls, cls.get_type()))?;
            for key in namespace.try_iter()? {
                hostile |= type_address(&key?) != s.str_;
            }
            namespaces.push(namespace);
        }
        let mut dict_get = None;
        if !hostile {
            for namespace in namespaces {
                let descriptor =
                    namespace.call_method1(intern!(py, "get"), (intern!(py, "__dict__"),))?;
                if type_address(&descriptor) == s.getset_descriptor {
                    // `type.__doc__` or `__annotations__` planted under this
                    // name would call its class-dict value's `__get__`.
                    let name = descriptor.getattr(intern!(py, "__name__"))?;
                    if type_address(&name) == s.str_ && name.eq(intern!(py, "__dict__"))? {
                        dict_get = Some(descriptor.getattr(intern!(py, "__get__"))?.unbind());
                    } else {
                        hostile = true;
                    }
                    break;
                } else if !descriptor.is_none() {
                    break;
                }
            }
        }
        let kind = match (scalar, base) {
            (Some(scalar), _) => Kind::Scalar(scalar),
            (None, Some(base)) => Kind::Container(base),
            (None, None) => Kind::Object,
        };
        Ok(Self {
            ty: ty.clone().unbind(),
            kind,
            ordered,
            dict_get,
            layout: None,
            hostile,
        })
    }

    /// Visits the Python objects held, for the garbage collector.
    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.ty)?;
        if let Some(getter) = &self.dict_get {
            visit.call(getter)?;
        }
        if let Some(layout) = &self.layout {
            for (name, getter) in &layout.members {
                visit.call(name)?;
                visit.call(getter)?;
            }
            for field in &layout.fields {
                visit.call(field)?;
            }
        }
        Ok(())
    }
}

/// Bound `__get__` of the instance-dict descriptor of `cls`, a copy class:
/// the first `__dict__` along its MRO, read at import, before application
/// code could give the class another.
pub(crate) fn dict_getter<'py>(cls: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = cls.py();
    for base in cls.getattr(intern!(py, "__mro__"))?.try_iter()? {
        let namespace = base?.getattr(intern!(py, "__dict__"))?;
        let descriptor = namespace.call_method1(intern!(py, "get"), (intern!(py, "__dict__"),))?;
        if !descriptor.is_none() {
            return descriptor.getattr(intern!(py, "__get__"));
        }
    }
    Err(PyTypeError::new_err("a copy class has no instance dict"))
}

/// The instance dict of `made`, a copy the walker built, through `getter`,
/// from [`dict_getter`]: the descriptor its class had at import, with no
/// hook application code gave the class since.
pub(crate) fn fresh_dict<'py>(
    getter: &Py<PyAny>,
    made: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    Ok(getter
        .bind(made.py())
        .call1((made,))?
        .cast_into::<PyDict>()?)
}

/// Whether reading the attribute `name` of an instance of `ty` looks first
/// in the instance dict, and runs no code while it finds it there: every key
/// of a class dict along the MRO is an exact str, and no class holds `name`
/// or a `__getattribute__` but `object`'s. (A `__getattr__` runs only for a
/// name the instance dict lacks.)
pub(crate) fn reads_plainly(
    py: Python<'_>,
    s: &Statics,
    ty: &Bound<'_, PyAny>,
    name: &Bound<'_, PyString>,
) -> PyResult<bool> {
    let mro = s.class_mro.bind(py).call1((ty, ty.get_type()))?;
    for cls in mro.try_iter()? {
        let cls = cls?;
        // Iterating a class dict yields its keys and runs nothing else.
        let namespace = s.class_dict.bind(py).call1((&cls, cls.get_type()))?;
        for key in namespace.try_iter()? {
            if type_address(&key?) != s.str_ {
                return Ok(false);
            }
        }
        // An entry of any value, `None` too, is the class's own.
        let object = s.object_getattribute.bind(py);
        let getattribute = namespace.call_method1(
            intern!(py, "get"),
            (intern!(py, "__getattribute__"), object),
        )?;
        if !getattribute.is(object) || namespace.contains(name)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The `reason` a `TruncatedValue`'s instance dict holds, if it is an exact
/// str in a dict of exact str keys, read through the descriptor the class
/// had at import, as attribute lookup reads it: reading it so runs no code
/// and compares only strs. `None` for any other.
pub(crate) fn plain_reason<'py>(
    py: Python<'py>,
    s: &Statics,
    placeholder: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let attributes = s.truncated_dict.bind(py).call1((placeholder,))?;
    // A dict, or a dict subclass assigned as `__dict__`, which attribute
    // lookup reads as a dict too.
    let attributes = attributes.cast_into::<PyDict>()?;
    if !s.str_keys(&attributes) {
        return Ok(None);
    }
    let reason = attributes.get_item(intern!(py, "reason"))?;
    Ok(reason.filter(|reason| type_address(reason) == s.str_))
}

/// One of pydumpling's classes as it was at import: the classes of its
/// MRO, which assigning `__bases__` changes, and their dicts but `object`'s,
/// read through `type.__dict__`. They held only exact str keys then, and
/// `setattr` adds any name as one, so while the MRO holds the same classes,
/// looking a name up there runs no code.
pub(crate) struct OwnClass {
    cls: Py<PyAny>,
    mro: Vec<Py<PyAny>>,
    namespaces: Vec<Py<PyAny>>,
}

impl OwnClass {
    /// Reads `cls` through `class_mro` and `class_dict`, the bound `__get__`
    /// of `type.__dict__["__mro__"]` and `["__dict__"]`.
    pub(crate) fn new(
        cls: &Bound<'_, PyAny>,
        class_mro: &Bound<'_, PyAny>,
        class_dict: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let py = cls.py();
        let mro = class_mro
            .call1((cls, cls.get_type()))?
            .cast_into::<PyTuple>()?;
        let mut namespaces = Vec::new();
        for base in &mro {
            if base.is(py.get_type::<PyAny>()) {
                continue;
            }
            let namespace = class_dict.call1((&base, base.get_type()))?;
            for key in namespace.try_iter()? {
                if !key?.is_exact_instance_of::<PyString>() {
                    return Err(PyTypeError::new_err("a class dict key is not a str"));
                }
            }
            namespaces.push(namespace.unbind());
        }
        let (cls, mro) = (
            cls.clone().unbind(),
            mro.iter().map(Bound::unbind).collect(),
        );
        Ok(Self {
            cls,
            mro,
            namespaces,
        })
    }

    /// Whether the class's MRO holds the classes it held at import.
    pub(crate) fn kept(&self, py: Python<'_>, s: &Statics) -> PyResult<bool> {
        let cls = self.cls.bind(py);
        let mro = s.class_mro.bind(py).call1((cls, cls.get_type()))?;
        let mro = mro.cast_into::<PyTuple>().map_err(PyErr::from)?;
        Ok(mro.len() == self.mro.len()
            && mro
                .iter()
                .zip(&self.mro)
                .all(|(now, then)| now.is(then.bind(py))))
    }

    /// What attribute access on an instance finds for each of `names` in
    /// the class dicts but `object`'s, `None` where none holds it, if the
    /// class keeps its MRO and none holds a `__getattribute__`: reading
    /// the instance dict or that entry then runs no hook.
    pub(crate) fn entries<'py, const N: usize>(
        &self,
        py: Python<'py>,
        s: &Statics,
        names: [&Bound<'py, PyString>; N],
    ) -> PyResult<Option<[Option<Bound<'py, PyAny>>; N]>> {
        if !self.kept(py, s)? {
            return Ok(None);
        }
        let mut found = std::array::from_fn(|_| None);
        for namespace in &self.namespaces {
            let namespace = namespace.bind(py);
            if namespace.contains(intern!(py, "__getattribute__"))? {
                return Ok(None);
            }
            for (name, entry) in names.iter().zip(found.iter_mut()) {
                if entry.is_none() {
                    // One lookup where the name is, as the caller expects.
                    *entry = match namespace.get_item(*name) {
                        Ok(held) => Some(held),
                        Err(err) if err.is_instance_of::<PyKeyError>(py) => None,
                        Err(err) => return Err(err),
                    };
                }
            }
        }
        Ok(Some(found))
    }
}

/// Whether `FakeClass.__repr__`, as the Python walk calls it on a copied
/// `FakeClass`, is the function the class had at import, and reading the
/// instance's `__dict__` there runs no hook.
pub(crate) fn plain_fake_class_repr(py: Python<'_>, s: &Statics) -> PyResult<bool> {
    let Some([Some(repr)]) = s.fake_class_own.entries(py, s, [intern!(py, "__repr__")])? else {
        return Ok(false);
    };
    Ok(repr.is(s.fake_class_repr.bind(py)))
}
