//! `ValuesMixin.value` and the copies it makes of builtin containers and
//! their subclasses' entries. Other objects are copied in
//! [`crate::objects`]; scalar subclasses and the attributes of subclass
//! copies in [`crate::copies`].
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

use crate::state::State;
use crate::statics::{address, id, type_address};
use crate::types::{Base, FakeClassNow, Kind, TypeInfo};
use crate::walker::{Halt, Walk};

/// Applies an `except Exception:` clause: `Ok(None)` for an `Exception`.
pub(crate) fn unless_exception<T>(py: Python<'_>, result: PyResult<T>) -> Walk<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(err) if err.is_instance_of::<PyException>(py) => Ok(None),
        Err(err) => Err(Halt::Py(err)),
    }
}

/// `dict.update(other)` for `other`, a dict or `FakeDict` the walk made.
pub(crate) fn update(target: &Bound<'_, PyDict>, other: &Bound<'_, PyAny>) -> PyResult<()> {
    target.update(other.cast::<PyDict>()?.as_mapping())
}

impl State {
    /// `ValuesMixin.value`: the memo lookup, then the node, natively or,
    /// below the native stack budget, in the Python walk in place.
    pub(crate) fn value<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
        depth: i64,
    ) -> Walk<Bound<'py, PyAny>> {
        if let Some(captured) = self.objects(py).get_item(id(value))? {
            return Ok(captured);
        }
        if self.stack_spent() {
            return self.python_value(py, value, depth);
        }
        self.fresh_value(py, value, depth)
    }

    /// `ValuesMixin.value` for a value the memo does not hold.
    pub(crate) fn fresh_value<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
        depth: i64,
    ) -> Walk<Bound<'py, PyAny>> {
        let s = self.s;
        let value_type = type_address(value);
        let exact_scalar = [
            s.truncated_type,
            s.none,
            s.bool_,
            s.float,
            s.complex,
            s.int,
            s.str_,
            s.bytes,
            s.bytearray,
        ]
        .contains(&value_type);
        // Read from the type, which runs no code, before the node counts.
        let kind = if exact_scalar {
            None
        } else {
            let kind = match TypeInfo::exact_kind(s, value_type) {
                Some(kind) => kind,
                None => self.info(py, &value.get_type())?.kind,
            };
            // Where the Python walk reads `_unlimited`: an unlimited capture
            // probes each value with the serializer.
            if !self.limited(py)?
                || (kind == Kind::Object && self.copies_in_python(py, value_type)?)
            {
                return self.elsewhere(py, value, depth);
            }
            Some(kind)
        };
        let id = &id(value);
        if !self.has_nodes(py)? {
            self.count(py, intern!(py, "max_nodes"))?;
            return Ok(self.node_marker(py)?);
        }
        self.add_node(py)?;
        if self.nodes.is_multiple_of(4096) {
            self.check_signals(py)?;
        }
        let Some(kind) = kind else {
            return self.exact_scalar(py, id, value, value_type);
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

    /// The value in the Python walk, as an unlimited context or a value only
    /// the Python walk copies takes it: in place, or, at the entry, before
    /// the call has done anything, by returning there, and for good if the
    /// context is unlimited. Out of line, as the frame of each native level
    /// would otherwise hold its locals.
    #[cold]
    #[inline(never)]
    fn elsewhere<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
        depth: i64,
    ) -> Walk<Bound<'py, PyAny>> {
        match (self.started, self.limited(py)?) {
            (true, _) => self.python_value(py, value, depth),
            (false, false) => Err(Halt::Retire),
            (false, true) => Err(Halt::InPython),
        }
    }

    /// The signal check, where a signal handler may capture through the
    /// context. Out of line, as [`State::elsewhere`].
    #[cold]
    #[inline(never)]
    fn check_signals(&mut self, py: Python<'_>) -> Walk<()> {
        self.application(py, || py.check_signals())
    }

    /// An exact `TruncatedValue`, `None`, bool, float, complex, int, str,
    /// bytes or bytearray, counted.
    fn exact_scalar<'py>(
        &mut self,
        py: Python<'py>,
        id: &Bound<'py, PyInt>,
        value: &Bound<'py, PyAny>,
        value_type: usize,
    ) -> Walk<Bound<'py, PyAny>> {
        let s = self.s;
        let captured = if value_type == s.int && self.int_too_long(py, value)? {
            self.omit(py, intern!(py, "max_string_bytes"), value)?
        } else if value_type == s.str_ {
            self.text(py, value)?
        } else if value_type == s.bytes || value_type == s.bytearray {
            self.bounded_bytes(py, value)?
        } else {
            value.clone()
        };
        self.remember(py, id, value, captured)
    }

    /// Whether an object of type `value_type` is copied in the Python walk:
    /// while `FakeClass` has other bases, or, for a `FakeClass` itself, a
    /// `__repr__` of other code than at import.
    fn copies_in_python(&mut self, py: Python<'_>, value_type: usize) -> Walk<bool> {
        Ok(match self.fake_class(py)? {
            FakeClassNow::Rebased => true,
            FakeClassNow::OtherRepr => value_type == self.s.fake_class_type,
            FakeClassNow::Hooked | FakeClassNow::Plain => false,
        })
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
        if !self.has_nodes(py)? && !self.objects(py).contains(id(item))? {
            self.count(py, intern!(py, "max_nodes"))?;
            return Ok(false);
        }
        let converted = self.value(py, item, depth + 1)?;
        match captured.cast::<PyList>() {
            Ok(list) => list.append(converted)?,
            Err(_) => captured
                .cast::<PySet>()
                .map_err(PyErr::from)?
                .add(converted)?,
        }
        *count += 1;
        Ok(true)
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
