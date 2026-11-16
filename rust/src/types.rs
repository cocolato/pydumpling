//! What the walk learns about a type from its MRO, cached per type.
//!
//! The Python walk rereads a value's MRO through `type.__dict__`
//! descriptors for every value. The walker reads it once per type until
//! application code runs ([`crate::state::State::forget_types`]), and keeps
//! the type alive so its address is not reused. A class changed in between
//! by Python code the walker does not sync around (a finalizer as it lets a
//! value go, or a collection or signal handler inside
//! `FakeContainer.__hash__`) is seen once the walker reads the type anew, an
//! accepted difference (`rust/DESIGN.md`).

use std::sync::Arc;

use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple, PyType};

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

/// Member descriptors and named tuple fields (`ValuesMixin._layout`), as
/// `ctx._layouts` holds them for a type.
pub(crate) enum Layout {
    /// What `_layout` writes: each slot's name and the bound `__get__` of
    /// its member descriptor, and the named tuple fields, exact strs.
    Plain {
        members: Vec<(Py<PyAny>, Py<PyAny>)>,
        fields: Vec<Py<PyAny>>,
    },
    /// Anything else `_layouts` holds for the type: its members and its
    /// fields, which the walk reads as the Python walk does, running their
    /// own code where they have it.
    Other {
        members: Py<PyAny>,
        fields: Py<PyAny>,
    },
}

/// The instance-dict descriptor `_attributes` finds for a type: the first
/// `__dict__` along its MRO, if a getset.
pub(crate) enum InstanceDict {
    /// The first `__dict__` is not a getset, or there is none.
    Absent,
    /// A getset named `__dict__`, which reads the instance dict and runs no
    /// code: its bound `__get__`.
    Plain(Py<PyAny>),
    /// Another getset planted as `__dict__` (`type.__doc__`, ...), whose
    /// getter may call application code: its bound `__get__`.
    Other(Py<PyAny>),
    /// A class dict along the MRO holds a key that is not an exact str,
    /// whose `__eq__` each lookup of `__dict__` may run: `_attributes`
    /// walks the MRO again for every value.
    Hostile,
}

/// The cached facts about one type.
pub(crate) struct TypeInfo {
    /// Keeps the type, and with it its address, alive for the call.
    _ty: Py<PyType>,
    pub kind: Kind,
    /// Whether `OrderedDict` is in the MRO (`_derives`).
    pub ordered: bool,
    pub dict: InstanceDict,
    /// The layout `_layouts` holds for the type, if `_layout` wrote it.
    pub layout: Option<Arc<Layout>>,
    /// Whether a class dict along the MRO has a key that is not an exact
    /// str, whose own `__eq__` or `__hash__` a lookup or set may run.
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
        let mut scalar = None;
        let mut base = None;
        let mut ordered = false;
        let mut hostile = false;
        let classes = class_dicts(s, ty.as_any())?;
        for (cls, namespace) in &classes {
            let cls_address = address(cls);
            scalar = scalar.or_else(|| s.scalar_of(cls_address));
            base = base.or_else(|| s.base_of(cls_address));
            ordered |= cls_address == s.ordered_dict;
            // Iterating a class dict yields its keys and runs nothing else.
            for key in namespace.try_iter()? {
                hostile |= type_address(&key?) != s.str_;
            }
        }
        let dict = match hostile {
            true => InstanceDict::Hostile,
            false => match dict_descriptor(py, classes.iter().map(|(_, namespace)| namespace))? {
                Some(descriptor) if type_address(&descriptor) == s.getset_descriptor => {
                    // `type.__doc__` or `__annotations__` planted under this
                    // name would call its class-dict value's `__get__`.
                    let name = descriptor.getattr(intern!(py, "__name__"))?;
                    let getter = descriptor.getattr(intern!(py, "__get__"))?.unbind();
                    if type_address(&name) == s.str_ && name.eq(intern!(py, "__dict__"))? {
                        InstanceDict::Plain(getter)
                    } else {
                        InstanceDict::Other(getter)
                    }
                }
                _ => InstanceDict::Absent,
            },
        };
        let kind = match (scalar, base) {
            (Some(scalar), _) => Kind::Scalar(scalar),
            (None, Some(base)) => Kind::Container(base),
            (None, None) => Kind::Object,
        };
        Ok(Self {
            _ty: ty.clone().unbind(),
            kind,
            ordered,
            dict,
            layout: None,
            hostile,
        })
    }
}

/// The classes along the MRO of `cls`, each with its class dict, read
/// through the `type.__dict__` descriptors, without any metaclass hook.
#[allow(clippy::type_complexity)]
pub(crate) fn class_dicts<'py>(
    s: &Statics,
    cls: &Bound<'py, PyAny>,
) -> PyResult<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    let py = cls.py();
    let (class_mro, class_dict) = (s.class_mro.bind(py), s.class_dict.bind(py));
    let mut found = Vec::new();
    for base in class_mro.call1((cls, cls.get_type()))?.try_iter()? {
        let base = base?;
        let namespace = class_dict.call1((&base, base.get_type()))?;
        found.push((base, namespace));
    }
    Ok(found)
}

/// The first `__dict__` the class dicts of an MRO hold, as `_attributes`
/// finds it with `.get("__dict__")` on each.
pub(crate) fn dict_descriptor<'a, 'py: 'a>(
    py: Python<'py>,
    namespaces: impl IntoIterator<Item = &'a Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let get = intern!(py, "get");
    for namespace in namespaces {
        let descriptor = namespace.call_method1(get, (intern!(py, "__dict__"),))?;
        if !descriptor.is_none() {
            return Ok(Some(descriptor));
        }
    }
    Ok(None)
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

/// `FakeClass` as it is now, as far as copying an object into it goes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FakeClassNow {
    /// Its MRO is not the one it had at import: it builds its copies
    /// through other bases, and 3.12 to 3.14 refuse the instance-dict
    /// descriptor it had then, so its objects are the Python walk's.
    Rebased,
    /// Its `__repr__`, which the Python walk calls from Python on a copied
    /// `FakeClass`, is not the function it had at import.
    OtherRepr,
    /// As at import, but a class along its MRO holds a `__getattribute__`,
    /// which that function's `self.__dict__` runs.
    Hooked,
    /// As at import.
    Plain,
}

impl FakeClassNow {
    /// Reads `FakeClass` along its MRO, running no code: while its MRO holds
    /// the classes it held at import, whose dicts held only exact str keys
    /// then (`setattr` adds any name as one), looking a name up there runs
    /// none.
    pub(crate) fn read<'py>(py: Python<'py>, s: &Statics) -> PyResult<Self> {
        let cls = s.fake_class.bind(py);
        let mro = s.class_mro.bind(py).call1((cls, cls.get_type()))?;
        let mro = mro.cast_into::<PyTuple>().map_err(PyErr::from)?;
        let then = &s.fake_class_mro;
        if mro.len() != then.len()
            || !mro
                .iter()
                .zip(then)
                .all(|(now, then)| now.is(then.bind(py)))
        {
            return Ok(FakeClassNow::Rebased);
        }
        // The first definition along the MRO, as attribute lookup takes it.
        let find = |name: &Bound<'py, PyString>| -> PyResult<Option<Bound<'py, PyAny>>> {
            for namespace in &s.fake_class_dicts {
                match namespace.bind(py).get_item(name) {
                    Ok(held) => return Ok(Some(held)),
                    Err(err) if err.is_instance_of::<PyKeyError>(py) => {}
                    Err(err) => return Err(err),
                }
            }
            Ok(None)
        };
        Ok(match find(intern!(py, "__repr__"))? {
            Some(repr) if repr.is(s.fake_class_repr.bind(py)) => {
                match find(intern!(py, "__getattribute__"))? {
                    Some(_) => FakeClassNow::Hooked,
                    None => FakeClassNow::Plain,
                }
            }
            _ => FakeClassNow::OtherRepr,
        })
    }
}
