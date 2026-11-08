//! Whether a context still runs the stock walk and keeps its state where
//! the walker reads it.
//!
//! The Python walk calls `self.value(...)`, `self._omit(...)` and the rest
//! of `values._NATIVE_WALK`, and reads and writes its state,
//! `values._NATIVE_STATE`, through the class's `__getattribute__`,
//! `__setattr__` and `__delattr__`; copies and pickles read the context
//! through `values._COPY_PROTOCOL`, on the instance, its class, and for
//! the class lookups of `copy.copy` and `copy.replace` the metaclass. A
//! class attribute of any of these names set, replaced or removed after the
//! context was made, such as a property in place of a state attribute, a
//! slot for `_walker`, which the instance releases before its dict, or a
//! reducer of the class's own, an instance attribute of a walk or copy
//! name, or a metaclass `__copy__`, `__replace__`, `__getattr__` or
//! `__getattribute__` but `type`'s, changes what the Python walk runs or
//! what a copy reads, and the walker then gives the context to it for good.
//! Otherwise the stock attribute access reads the state from the instance
//! dict, and so does the walker, without any hook.
//!
//! The check runs before every call and runs no application code. Class
//! dicts are read through the `type.__dict__` descriptors, never through a
//! metaclass, and a lookup finds the raw class attribute, as an instance
//! attribute lookup does, without calling its `__get__`. Only the classes
//! in front of each name's owner are searched, and no immutable builtin
//! class such as `object`, which nothing can change. A metaclass dict, like
//! a class dict, holds only exact str keys, whose lookup runs no `__eq__`.

use pyo3::exceptions::{PyException, PyKeyError};
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};
use pyo3::PyTraverseError;

use crate::statics::{type_address, Statics};
use crate::types::TypeInfo;

/// `Py_TPFLAGS_IMMUTABLETYPE`: a class whose attributes cannot be set.
const IMMUTABLE_TYPE: u64 = 1 << 8;

/// What a context type held, along its MRO, when last checked.
pub(crate) struct ContextType {
    ty: Py<PyType>,
    /// The MRO tuple: assigning `__bases__` replaces it.
    mro: Py<PyAny>,
    /// Each MRO class's dict, read through its mapping proxy, or `None`
    /// for an immutable class, which the check skips.
    namespaces: Vec<Option<Py<PyAny>>>,
    /// For each of [`Statics::walk_keys`], the MRO index of the class that
    /// holds it and the attribute it held, or `None` if no class did.
    /// Empty unless every name resolved as on the stock class.
    owners: Vec<Option<(usize, Py<PyAny>)>>,
    /// Whether every name resolved as on the stock class when computed.
    stock: bool,
    /// The metaclass, its MRO and the dicts of its classes but `type` and
    /// `object`, where none of [`Statics::meta_keys`] may appear; `None`
    /// for `type`, which no class can trade for another.
    meta: Option<Meta>,
    /// Bound `__get__` of the instance-dict descriptor, if the type has a
    /// plain one and no class dict key whose own methods a lookup may run.
    dict_get: Option<Py<PyAny>>,
}

impl ContextType {
    /// Reads `ty`'s MRO and compares each name with `stock`, the class
    /// attributes `values._walk_methods` found on the stock context class.
    fn compute(
        py: Python<'_>,
        s: &Statics,
        ty: &Bound<'_, PyType>,
        stock: &Bound<'_, PyTuple>,
    ) -> PyResult<Self> {
        let (mro, namespaces) = class_dicts(py, s, ty)?;
        let (metaclass, types) = (ty.get_type(), py.get_type::<PyType>());
        let mut plain_meta = true;
        let meta = match metaclass.is(&types) {
            true => None,
            false => {
                let (meta_mro, classes) = class_dicts(py, s, &metaclass)?;
                let mut dicts = Vec::new();
                for (cls, namespace, _) in classes {
                    if cls.is(&types) || cls.is(py.get_type::<PyAny>()) {
                        continue;
                    }
                    for key in namespace.try_iter()? {
                        plain_meta &= type_address(&key?) == s.str_;
                    }
                    dicts.push(namespace.unbind());
                }
                Some((metaclass.unbind(), meta_mro.unbind(), dicts))
            }
        };
        let info = TypeInfo::compute(py, s, ty)?;
        // In a class dict with a key other than an exact str, looking a
        // name up may run that key's own `__eq__`: the Python walk must.
        let mut stock_names = !info.hostile
            && plain_meta
            && stock.len() == s.walk_keys.len()
            && !meta_answers(py, s, &meta)?;
        let mut owners = Vec::with_capacity(s.walk_keys.len());
        for (index, name) in s.walk_keys.iter().enumerate() {
            if !stock_names {
                break;
            }
            let name = name.bind(py);
            let mut owner = None;
            for (at, (_, namespace, _)) in namespaces.iter().enumerate() {
                if namespace.contains(name)? {
                    owner = Some((at, namespace.get_item(name)?.unbind()));
                    break;
                }
            }
            let expected = stock.get_item(index)?;
            stock_names = match &owner {
                Some((_, found)) => found.bind(py).is(&expected),
                None => expected.is_none(),
            };
            owners.push(owner);
        }
        if !stock_names {
            owners.clear();
        }
        Ok(Self {
            ty: ty.clone().unbind(),
            mro: mro.unbind(),
            namespaces: namespaces
                .into_iter()
                .map(|(_, namespace, mutable)| mutable.then(|| namespace.unbind()))
                .collect(),
            owners,
            stock: stock_names,
            meta,
            dict_get: info.dict_get.filter(|_| !info.hostile),
        })
    }

    /// Whether every class still holds what it held when computed.
    fn unchanged(&self, py: Python<'_>, s: &Statics) -> PyResult<bool> {
        if let Some((metaclass, mro, _)) = &self.meta {
            let ty = self.ty.bind(py);
            let current = s
                .class_mro
                .bind(py)
                .call1((metaclass, metaclass.bind(py).get_type()))?;
            if !ty.get_type().is(metaclass) || !current.is(mro) || meta_answers(py, s, &self.meta)?
            {
                return Ok(false);
            }
        }
        for (name, owner) in s.walk_keys.iter().zip(&self.owners) {
            let name = name.bind(py);
            let (front, held) = match owner {
                Some((at, held)) => (&self.namespaces[..*at], Some((*at, held))),
                None => (&self.namespaces[..], None),
            };
            for namespace in front.iter().flatten() {
                if namespace.bind(py).contains(name)? {
                    return Ok(false);
                }
            }
            let Some((at, held)) = held else {
                continue;
            };
            let Some(namespace) = &self.namespaces[at] else {
                continue;
            };
            match namespace.bind(py).get_item(name) {
                Ok(found) if found.is(held.bind(py)) => {}
                Ok(_) => return Ok(false),
                Err(err) if err.is_instance_of::<PyKeyError>(py) => return Ok(false),
                Err(err) => return Err(err),
            }
        }
        Ok(true)
    }

    /// The context's own dict, if the type is the one computed and the
    /// dict leaves every walk method to its class: an instance attribute
    /// is what the Python walk would call.
    fn attributes<'py>(
        &self,
        py: Python<'py>,
        s: &Statics,
        ctx: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(getter) = &self.dict_get else {
            return Ok(None);
        };
        if !ctx.get_type().is(self.ty.bind(py)) {
            return Ok(None);
        }
        // A `__dict__` getset another class defined refuses this instance:
        // the Python walk reads the state its own way.
        let attributes = match getter.bind(py).call1((ctx, self.ty.bind(py))) {
            Ok(attributes) => attributes,
            Err(err) if err.is_instance_of::<PyException>(py) => return Ok(None),
            Err(err) => return Err(err),
        };
        let Ok(attributes) = attributes.cast_into_exact::<PyDict>() else {
            return Ok(None);
        };
        // Looking a str up compares it with other keys; any key but an
        // exact str may run its own `__eq__`.
        for (key, _) in attributes.iter() {
            if type_address(&key) != s.str_ {
                return Ok(None);
            }
        }
        let names = s.walk_keys[..s.walk_count].iter();
        for name in names.chain(&s.walk_keys[s.copy_from..]) {
            if attributes.contains(name.bind(py))? {
                return Ok(None);
            }
        }
        Ok(Some(attributes))
    }

    /// Visits the Python objects held, for the garbage collector.
    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.ty)?;
        visit.call(&self.mro)?;
        for namespace in self.namespaces.iter().flatten() {
            visit.call(namespace)?;
        }
        for (_, held) in self.owners.iter().flatten() {
            visit.call(held)?;
        }
        if let Some(getter) = &self.dict_get {
            visit.call(getter)?;
        }
        if let Some((metaclass, mro, dicts)) = &self.meta {
            visit.call(metaclass)?;
            visit.call(mro)?;
            for namespace in dicts {
                visit.call(namespace)?;
            }
        }
        Ok(())
    }
}

/// A metaclass, its MRO and its classes' dicts but `type`'s and `object`'s.
type Meta = (Py<PyType>, Py<PyAny>, Vec<Py<PyAny>>);

/// Each class in an MRO, its dict, and whether the class is mutable.
type Dicts<'py> = Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>, bool)>;

/// The MRO of `cls`, and each class with its dict, read through
/// `type.__dict__`'s descriptors, and whether that class is mutable.
fn class_dicts<'py>(
    py: Python<'py>,
    s: &Statics,
    cls: &Bound<'py, PyType>,
) -> PyResult<(Bound<'py, PyAny>, Dicts<'py>)> {
    let mro = s.class_mro.bind(py).call1((cls, cls.get_type()))?;
    let mut namespaces = Vec::new();
    for base in mro.try_iter()? {
        let base = base?;
        let namespace = s.class_dict.bind(py).call1((&base, base.get_type()))?;
        let mutable = mutable(py, s, &base)?;
        namespaces.push((base, namespace, mutable));
    }
    Ok((mro, namespaces))
}

/// Whether `cls`, a class, is one whose attributes can be set.
fn mutable(py: Python<'_>, s: &Statics, cls: &Bound<'_, PyAny>) -> PyResult<bool> {
    let flags: u64 = s
        .class_flags
        .bind(py)
        .call1((cls, cls.get_type()))?
        .extract()?;
    Ok(flags & IMMUTABLE_TYPE == 0)
}

/// Whether a metaclass class dict holds one of [`Statics::meta_keys`].
fn meta_answers(py: Python<'_>, s: &Statics, meta: &Option<Meta>) -> PyResult<bool> {
    for namespace in meta.iter().flat_map(|(_, _, dicts)| dicts) {
        for name in &s.meta_keys {
            if namespace.bind(py).contains(name.bind(py))? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// The instance dict of `ctx`, the state the walker mirrors, if `ctx` runs
/// the stock walk methods; `cached` is reused while the context's type and
/// MRO stay the same.
pub(crate) fn unshadowed<'py>(
    py: Python<'py>,
    s: &Statics,
    cached: &mut Option<ContextType>,
    ctx: &Bound<'py, PyAny>,
    stock: &Bound<'_, PyTuple>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let ty = ctx.get_type();
    let known = match cached {
        Some(known) if known.ty.bind(py).is(&ty) => {
            let mro = s.class_mro.bind(py).call1((&ty, ty.get_type()))?;
            mro.is(known.mro.bind(py)) && known.unchanged(py, s)?
        }
        _ => false,
    };
    if !known {
        *cached = Some(ContextType::compute(py, s, &ty, stock)?);
    }
    match cached.as_ref() {
        Some(current) if current.stock => current.attributes(py, s, ctx),
        _ => Ok(None),
    }
}

/// The instance dict of `ctx` through the type last checked, as after a
/// hand-back, without checking the classes again.
pub(crate) fn attributes<'py>(
    py: Python<'py>,
    s: &Statics,
    cached: &Option<ContextType>,
    ctx: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    match cached {
        Some(current) if current.stock => current.attributes(py, s, ctx),
        _ => Ok(None),
    }
}
