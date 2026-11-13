//! `ValuesMixin.value` and the copies it makes of builtin containers, their
//! subclasses' entries, and other objects as `FakeClass`; scalar subclasses
//! and the attributes of subclass copies are in [`crate::copies`].
//!
//! Every step follows the Python walk in order, because the order of memo
//! entries, node counts, and truncation counters is part of the snapshot;
//! the memo is the context's own `objects`, read and written as the Python
//! walk reads and writes it.

use std::collections::hash_map;

use pyo3::exceptions::PyException;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList, PySet, PyString, PyTuple, PyType};

use crate::statics::{address, id, type_address};
use crate::types::{fresh_dict, plain_fake_class_repr, Base, Kind, TypeInfo};
use crate::walker::{Halt, State, Walk};

/// Applies an `except Exception:` clause: `Ok(None)` for an `Exception`.
pub(crate) fn unless_exception<T>(py: Python<'_>, result: PyResult<T>) -> Walk<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(err) => match Halt::from(err) {
            Halt::Py(err) if err.is_instance_of::<PyException>(py) => Ok(None),
            halt => Err(halt),
        },
    }
}

/// `dict.update(other)` for a dict or dict subclass `other`.
pub(crate) fn update(target: &Bound<'_, PyDict>, other: &Bound<'_, PyAny>) -> Walk<()> {
    match other.cast::<PyDict>() {
        Ok(other) => target.update(other.as_mapping())?,
        Err(_) => {
            target.call_method1(intern!(target.py(), "update"), (other,))?;
        }
    }
    Ok(())
}

impl State {
    /// `ValuesMixin.value`.
    pub(crate) fn value<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
        depth: i64,
    ) -> Walk<Bound<'py, PyAny>> {
        self.enter(py)?;
        let captured = self.value_body(py, value, depth);
        self.leave();
        captured
    }

    fn value_body<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
        depth: i64,
    ) -> Walk<Bound<'py, PyAny>> {
        let s = self.s;
        let id = &id(value);
        if let Some(captured) = self.objects(py)?.get_item(id)? {
            return Ok(captured);
        }
        if !self.has_nodes() {
            self.count(py, intern!(py, "max_nodes"))?;
            return Ok(self.node_marker.bind(py).clone());
        }
        self.nodes += 1;
        if self.nodes.is_multiple_of(4096) {
            py.check_signals()?;
        }
        let value_type = type_address(value);
        if value_type == s.truncated_type
            || value_type == s.none
            || value_type == s.bool_
            || value_type == s.float
            || value_type == s.complex
        {
            return self.remember(py, id, value, value.clone());
        }
        if value_type == s.int {
            if self.int_too_long(py, value)? {
                let omitted = self.omit(py, intern!(py, "max_string_bytes"), value)?;
                return self.remember(py, id, value, omitted);
            }
            return self.remember(py, id, value, value.clone());
        }
        if value_type == s.str_ {
            let text = self.text(py, value)?;
            return self.remember(py, id, value, text);
        }
        if value_type == s.bytes || value_type == s.bytearray {
            let bytes = self.bounded_bytes(py, value)?;
            return self.remember(py, id, value, bytes);
        }
        let kind = match TypeInfo::exact_kind(s, value_type) {
            Some(kind) => kind,
            None => self.info(py, &value.get_type())?.kind,
        };
        if let Kind::Scalar(scalar) = kind {
            // Like an exact scalar, the value is kept at any depth.
            return self.scalar(py, id, value, scalar, depth);
        }
        if self.depth_reached(depth) {
            let omitted = self.omit(py, intern!(py, "max_depth"), value)?;
            return self.remember(py, id, value, omitted);
        }
        match kind {
            Kind::Container(base) => self.container(py, id, value, base, depth),
            _ => self.object(py, id, value, depth),
        }
    }

    /// Whether an exact int is longer than the string budget.
    fn int_too_long(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> Walk<bool> {
        let Some(limit) = self.max_string_bytes else {
            return Ok(false);
        };
        let budget = u128::from(limit) * 8;
        if budget >= 64 && value.extract::<i64>().is_ok() {
            return Ok(false);
        }
        let bits: u64 = self.s.int_bit_length.bind(py).call1((value,))?.extract()?;
        Ok(u128::from(bits) > budget)
    }

    /// The cached facts about `ty`, computed on first use.
    pub(crate) fn info(&mut self, py: Python<'_>, ty: &Bound<'_, PyType>) -> Walk<&mut TypeInfo> {
        let s = self.s;
        Ok(match self.types.entry(address(ty.as_any())) {
            hash_map::Entry::Occupied(known) => known.into_mut(),
            hash_map::Entry::Vacant(slot) => slot.insert(TypeInfo::compute(py, s, ty)?),
        })
    }

    /// `ValuesMixin._attributes`: the instance dict, read only through
    /// Python's real instance-dictionary descriptor.
    pub(crate) fn attributes<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
    ) -> Walk<Option<Bound<'py, PyDict>>> {
        let ty = value.get_type();
        let info = self.info(py, &ty)?;
        if info.hostile {
            return Err(Halt::Bail);
        }
        let Some(getter) = info.dict_get.as_ref().map(|g| g.clone_ref(py)) else {
            return Ok(None);
        };
        let Some(attributes) = unless_exception(py, getter.bind(py).call1((value, &ty)))? else {
            return Ok(None);
        };
        if type_address(&attributes) != self.s.dict {
            return Ok(None);
        }
        Ok(attributes.cast_into::<PyDict>().ok())
    }

    /// A builtin container or a subclass instance of one.
    fn container<'py>(
        &mut self,
        py: Python<'py>,
        id: &Bound<'py, PyInt>,
        value: &Bound<'py, PyAny>,
        base: Base,
        depth: i64,
    ) -> Walk<Bound<'py, PyAny>> {
        let s = self.s;
        let (plain, snapshot) = s.copy_types(py, base);
        let exact = type_address(value) == address(&s.base_cls(py, base));
        // A subclass keeps its entries in a snapshot subclass of its base,
        // with its type name and attributes alongside.
        let target = if exact {
            plain
        } else {
            snapshot.cls.bind(py).clone()
        };
        let filled = target.is(snapshot.cls.bind(py));
        if matches!(base, Base::Dict | Base::FrozenDict) {
            let captured = self.mapping(py, value, depth, None, true, &target)?;
            if filled {
                self.fill_copy(py, value, &captured, snapshot, depth)?;
            }
            return Ok(captured);
        }
        let sequence = matches!(base, Base::List | Base::Tuple);
        let mutable = matches!(base, Base::List | Base::Set);
        let captured = match (mutable && !exact, sequence) {
            (true, _) => snapshot.make(py, None)?,
            (false, true) => PyList::empty(py).into_any(),
            (false, false) => PySet::empty(py)?.into_any(),
        };
        if mutable {
            self.remember(py, id, value, captured.clone())?;
        } else {
            // What a reference back to it finds until it is filled.
            let type_name = self.type_name(py, value)?;
            let reason = s.recursive_immutable.bind(py).clone();
            let placeholder = self.truncated(py, &reason, &type_name)?;
            self.remember(py, id, value, placeholder)?;
        }
        let mut count = 0;
        // Iterate through the base type: a subclass `__iter__` must be
        // neither executed nor able to hide entries.
        match base {
            Base::List => {
                let list = value.cast::<PyList>().map_err(PyErr::from)?;
                let mut index = 0;
                while index < list.len() {
                    let item = list.get_item(index)?;
                    if !self.take(py, &captured, &item, &mut count, depth)? {
                        break;
                    }
                    index += 1;
                }
            }
            Base::Tuple => {
                for item in value.cast::<PyTuple>().map_err(PyErr::from)?.iter() {
                    if !self.take(py, &captured, &item, &mut count, depth)? {
                        break;
                    }
                }
            }
            _ => {
                let iterator = if exact {
                    value.try_iter()?
                } else if base == Base::Set {
                    s.set_iter.bind(py).call1((value,))?.try_iter()?
                } else {
                    s.frozenset_iter.bind(py).call1((value,))?.try_iter()?
                };
                for item in iterator {
                    if !self.take(py, &captured, &item?, &mut count, depth)? {
                        break;
                    }
                }
            }
        }
        let captured = if mutable {
            captured
        } else if filled {
            let finished = snapshot.make(py, Some(&captured))?;
            self.remember(py, id, value, finished)?
        } else {
            let finished = target.call1((captured,))?;
            self.remember(py, id, value, finished)?
        };
        if filled {
            self.fill_copy(py, value, &captured, snapshot, depth)?;
        }
        Ok(captured)
    }

    /// Adds one sequence or set item; `false` once a budget stops the copy.
    fn take<'py>(
        &mut self,
        py: Python<'py>,
        captured: &Bound<'py, PyAny>,
        item: &Bound<'py, PyAny>,
        count: &mut u64,
        depth: i64,
    ) -> Walk<bool> {
        if self.max_items.is_some_and(|limit| *count >= limit) {
            self.count(py, intern!(py, "max_items"))?;
            return Ok(false);
        }
        if !self.has_nodes() && !self.objects(py)?.contains(id(item))? {
            self.count(py, intern!(py, "max_nodes"))?;
            return Ok(false);
        }
        let converted = self.value(py, item, depth + 1)?;
        match captured.cast::<PyList>() {
            Ok(list) => list.append(converted)?,
            Err(_) => {
                if !self.s.plain_key(&converted) {
                    // A copy's own `__hash__`, `FakeContainer`'s, is Python.
                    self.before_python(py)?;
                }
                captured
                    .cast::<PySet>()
                    .map_err(PyErr::from)?
                    .add(converted)?
            }
        }
        *count += 1;
        Ok(true)
    }

    /// A `FakeClass` with the object's converted attributes.
    fn object<'py>(
        &mut self,
        py: Python<'py>,
        id: &Bound<'py, PyInt>,
        value: &Bound<'py, PyAny>,
        depth: i64,
    ) -> Walk<Bound<'py, PyAny>> {
        let s = self.s;
        // A `FakeClass` given other bases builds its copies through theirs,
        // and before 3.15 refuses the instance-dict descriptor it had at
        // import: checked before anything runs that the Python walk reruns.
        if !s.fake_class_own.kept(py, s)? {
            return Err(Halt::Bail);
        }
        let fake_class = type_address(value) == s.fake_class_type;
        let representation = if fake_class {
            // The Python walk calls `FakeClass.__repr__` as the class has
            // it then, which looks its text up in the instance dict: here,
            // only the one it had at import, with no hook, in an exact dict
            // of exact str keys, which the attribute copy below needs too.
            // Reading the instance dict checks the class dicts' keys first.
            match self.attributes(py, value)? {
                Some(attrs) if s.str_keys(&attrs) => {}
                _ => return Err(Halt::Bail),
            }
            if !plain_fake_class_repr(py, s)? {
                return Err(Halt::Bail);
            }
            self.before_python(py)?;
            s.fake_class_repr.bind(py).call1((value,))?
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
}

/// Whether `value` is an exact `str`.
pub(crate) fn is_exact_str(state: &State, value: &Bound<'_, PyAny>) -> bool {
    type_address(value) == state.s.str_
}

/// Whether an exact str equals `text`, compared as Python's `==` would.
pub(crate) fn str_equals(value: &Bound<'_, PyAny>, text: &Bound<'_, PyString>) -> Walk<bool> {
    Ok(value.len()? == text.len()? && value.eq(text)?)
}
