//! What the walk learns about a type from its MRO, cached per type.
//!
//! The Python walk rereads a value's MRO through `type.__dict__`
//! descriptors for every value. The walker reads it once per type and
//! call, and keeps the type alive so its address is not reused. Native code
//! runs no application code; a class that code run inside a call changes
//! (a signal handler, a collection, another thread) is seen from the next
//! call, an accepted difference (`rust/DESIGN.md`).

use std::sync::Arc;

use pyo3::gc::PyVisit;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyType;
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

impl Layout {
    /// Converts `(members, fields)` from `ValuesMixin._layout`.
    pub(crate) fn from_python(py: Python<'_>, pair: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut members = Vec::new();
        for member in pair.get_item(0)?.try_iter()? {
            let member = member?;
            let getter = member.get_item(1)?.getattr(intern!(py, "__get__"))?;
            members.push((member.get_item(0)?.unbind(), getter.unbind()));
        }
        let fields = pair
            .get_item(1)?
            .try_iter()?
            .map(|field| field.map(Bound::unbind))
            .collect::<PyResult<_>>()?;
        Ok(Self { members, fields })
    }
}
