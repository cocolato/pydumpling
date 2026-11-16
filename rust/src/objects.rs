//! Objects, copied as `FakeClass`, and the instance dict that the copies
//! of objects and of subclass instances keep (`ValuesMixin._attributes`).

use pyo3::exceptions::PyException;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt};

use crate::state::State;
use crate::statics::type_address;
use crate::types::{class_dicts, dict_descriptor, fresh_dict, FakeClassNow, InstanceDict};
use crate::walk::{unless_exception, update};
use crate::walker::Walk;

impl State {
    /// `ValuesMixin._attributes`: the instance dict, read only through
    /// Python's real instance-dictionary descriptor, if an exact dict.
    pub(crate) fn attributes<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
    ) -> Walk<Option<Bound<'py, PyDict>>> {
        let ty = value.get_type();
        let found = match &self.info(py, &ty)?.dict {
            InstanceDict::Absent => None,
            InstanceDict::Plain(getter) => {
                let getter = getter.clone_ref(py);
                unless_exception(py, getter.bind(py).call1((value, &ty)))?
            }
            InstanceDict::Other(getter) => {
                // A getset planted as `__dict__` may run application code.
                let getter = getter.clone_ref(py);
                let got = self.application(py, || Ok(getter.bind(py).call1((value, &ty))))?;
                unless_exception(py, got)?
            }
            InstanceDict::Hostile => self.hostile_attributes(py, value)?,
        };
        Ok(found
            .filter(|attrs| type_address(attrs) == self.s.dict)
            .and_then(|attrs| attrs.cast_into::<PyDict>().ok()))
    }

    /// `_attributes` for a type along whose MRO a class dict holds a key
    /// that is not an exact str: the same reads, whose `.get("__dict__")`
    /// runs such a key's `__eq__` as the Python walk's does.
    fn hostile_attributes<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
    ) -> Walk<Option<Bound<'py, PyAny>>> {
        let s = self.s;
        self.application(py, || {
            let ty = value.get_type();
            let classes = class_dicts(s, ty.as_any())?;
            let found = dict_descriptor(py, classes.iter().map(|(_, namespace)| namespace))?;
            let Some(descriptor) = found.filter(|found| type_address(found) == s.getset_descriptor)
            else {
                return Ok(None);
            };
            let getter = descriptor.getattr(intern!(py, "__get__"))?;
            match getter.call1((value, ty)) {
                Ok(attrs) => Ok(Some(attrs)),
                Err(err) if err.is_instance_of::<PyException>(py) => Ok(None),
                Err(err) => Err(err),
            }
        })
    }

    /// A `FakeClass` with the object's converted attributes.
    pub(crate) fn object<'py>(
        &mut self,
        py: Python<'py>,
        id: &Bound<'py, PyInt>,
        value: &Bound<'py, PyAny>,
        depth: i64,
    ) -> Walk<Bound<'py, PyAny>> {
        let s = self.s;
        let fake_class = type_address(value) == s.fake_class_type;
        let representation = if fake_class {
            self.fake_class_repr(py, value)?
        } else {
            let type_name = self.type_name(py, value)?;
            self.snapshot_repr(py, &type_name)?
        };
        let private = intern!(py, "_FakeClass__repr");
        // `FakeClass(representation, {})`, built as its `__init__` would,
        // through the instance-dict descriptor the class had at import: no
        // hook application code gave the class runs.
        let captured = s.object_new.bind(py).call1((s.fake_class.bind(py),))?;
        let namespace = fresh_dict(&s.fake_class_dict, &captured)?;
        namespace.set_item(private, &representation)?;
        self.remember(py, id, value, captured.clone())?;
        match self.attributes(py, value)? {
            Some(attrs) => {
                let private = fake_class.then_some(private.as_any());
                let converted = self.attribute_mapping(py, attrs.as_any(), depth + 1, private)?;
                update(&namespace, &converted)?;
                namespace.set_item(intern!(py, "_FakeClass__repr"), &representation)?;
            }
            None => self.count(py, intern!(py, "unsupported_type"))?,
        }
        Ok(captured)
    }

    /// `FakeClass.__repr__(value)`, the function the class had at import:
    /// `self.__dict__.get("_FakeClass__repr", "<object snapshot>")`. An
    /// exact dict of exact str keys read through the real instance-dict
    /// descriptor, with no `__getattribute__` along the way, is read here;
    /// anything else runs the same lookups generically, with their code.
    fn fake_class_repr<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        let s = self.s;
        let (key, default) = (
            intern!(py, "_FakeClass__repr"),
            intern!(py, "<object snapshot>"),
        );
        let now = self.fake_class(py)?;
        let getter = match &self.info(py, &value.get_type())?.dict {
            InstanceDict::Plain(getter) if now == FakeClassNow::Plain => Some(getter.clone_ref(py)),
            _ => None,
        };
        let attrs = match getter {
            Some(getter) => getter.bind(py).call1((value, value.get_type()))?,
            None => self.application(py, || value.getattr(intern!(py, "__dict__")))?,
        };
        if let Ok(attrs) = attrs.cast_exact::<PyDict>() {
            if s.str_keys(attrs) {
                return Ok(attrs
                    .get_item(key)?
                    .unwrap_or_else(|| default.clone().into_any()));
            }
        }
        // `.get` of a dict subclass, or a lookup that compares keys of
        // other types, runs their code.
        self.application(py, || {
            attrs.call_method1(intern!(py, "get"), (key, default))
        })
    }
}
