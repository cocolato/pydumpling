//! `ValuesMixin._mapping` and `_attribute_mapping`: bounded mapping copies.
//!
//! Values are read through the builtin base's `items`, so no subclass
//! override or key hook runs; [`crate::keys`] selects which entries to copy.
//! Any other mapping's own `items` is application code, which the Python
//! walk reads up to three times: such a mapping, which only a caller of
//! `_mapping` passes, is the Python walk's from the entry. Only a
//! `frozendict` (3.15) and a frame's `f_locals` proxy, whose `items()` runs
//! no code, are read here too.

use pyo3::exceptions::{PyRecursionError, PyRuntimeError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet};

use crate::address::AddressMap;
use crate::keys::{pair, Chosen};
use crate::state::State;
use crate::statics::{address, id, type_address};
use crate::types::{Base, Kind};
use crate::walk::is_exact_str;
use crate::walker::{Halt, Walk};

impl State {
    /// `ValuesMixin._mapping` as its entry receives it: a source whose
    /// items the walker does not read, decided from its type before
    /// anything runs, is the Python walk's.
    pub(crate) fn mapping_entry<'py>(
        &mut self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        depth: i64,
        priority: &Bound<'py, PyAny>,
        counted: bool,
        container: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        if self.reader(py, source)?.is_none() {
            return Err(Halt::InPython);
        }
        self.mapping(py, source, depth, Some(priority), counted, container)
    }

    /// How `_mapping` reads the items of `source`, if the walker reads
    /// them: through its builtin base's `items`, or a frame locals proxy's
    /// own, which runs no code.
    fn reader(&mut self, py: Python<'_>, source: &Bound<'_, PyAny>) -> Walk<Option<(Kind, bool)>> {
        let s = self.s;
        let source_type = type_address(source);
        if source_type == s.dict {
            // A frame namespace, mostly: no type facts to read.
            return Ok(Some((Kind::Container(Base::Dict), false)));
        }
        let info = self.info(py, &source.get_type())?;
        Ok(match info.kind {
            Kind::Container(_) => Some((info.kind, info.ordered)),
            _ if source_type == s.frame_locals && s.frame_locals != 0 => {
                Some((Kind::Object, false))
            }
            _ => None,
        })
    }

    /// `ValuesMixin._mapping`.
    pub(crate) fn mapping<'py>(
        &mut self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        depth: i64,
        priority: Option<&Bound<'py, PyAny>>,
        counted: bool,
        container: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        let s = self.s;
        let id = &id(source);
        // Looked up twice, as the Python walk's `in` and subscript do.
        if self.objects(py).contains(id)? {
            let cached = self.objects(py).as_any().get_item(id)?;
            let cached_type = type_address(&cached);
            if cached_type == s.dict || cached_type == s.fake_dict_type {
                return Ok(cached);
            }
            // A mapping root may fit even when a deeper reference was
            // omitted: `not isinstance(cached, TruncatedValue) or
            // cached.reason != "max_depth"`, which reads a builtin value's
            // class running no code, and any other entry's `__class__`, a
            // placeholder's `reason` and its `!=` with their own code.
            let omitted = if s.plain_keys.contains(&cached_type)
                || [s.bytearray, s.list, s.tuple, s.set, s.frozenset].contains(&cached_type)
            {
                true
            } else {
                let truncated = s.truncated.bind(py);
                self.application(py, || {
                    Ok(!cached.is_instance(truncated)?
                        || cached
                            .getattr(intern!(py, "reason"))?
                            .ne(intern!(py, "max_depth"))?)
                })?
            };
            if omitted {
                self.count(py, intern!(py, "unsupported_type"))?;
                return Ok(PyDict::new(py).into_any());
            }
        }
        if !counted {
            if !self.has_nodes() {
                self.count(py, intern!(py, "max_nodes"))?;
                return Ok(PyDict::new(py).into_any());
            }
            self.nodes += 1;
        }
        if self.depth_reached(depth) {
            self.count(py, intern!(py, "max_depth"))?;
            let empty = self.s.empty_mapping(py, container)?;
            return self.remember(py, id, source, empty);
        }
        let empty = self.s.empty_mapping(py, container)?;
        let captured = self.remember(py, id, source, empty)?;
        // Frame variable names are identifiers, not application string
        // values, so a string budget must not rewrite them.
        let names = self.named(py, id)?;
        // `set(priority)`, of exact strs; an empty one selects nothing.
        let preferred = match priority {
            Some(priority) if priority.len()? > 0 => {
                let preferred = PySet::empty(py)?;
                for name in priority.try_iter()? {
                    preferred.add(name?)?;
                }
                Some(preferred)
            }
            _ => None,
        };
        match self.fill_mapping(py, source, depth, &captured, names, preferred.as_ref()) {
            Ok(()) => {}
            Err(Halt::Py(err))
                if err.is_instance_of::<PyRuntimeError>(py)
                    && !err.is_instance_of::<PyRecursionError>(py) =>
            {
                self.count(py, intern!(py, "concurrent_mutation"))?;
            }
            Err(halt) => return Err(halt),
        }
        Ok(captured)
    }

    /// The body of `_mapping`'s `try` statement.
    fn fill_mapping<'py>(
        &mut self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        depth: i64,
        captured: &Bound<'py, PyAny>,
        names: bool,
        preferred: Option<&Bound<'py, PySet>>,
    ) -> Walk<()> {
        let s = self.s;
        let Some((kind, mut ordered)) = self.reader(py, source)? else {
            unreachable!("the entry and the walk's own calls pass only mappings it reads");
        };
        let view = match kind {
            Kind::Container(Base::Dict) => s.dict_items.bind(py).call1((source,))?,
            Kind::Container(Base::FrozenDict) => match &s.frozendict_items {
                Some(items) => items.bind(py).call1((source,))?,
                None => unreachable!("frozendict only exists where its items does"),
            },
            Kind::Container(base) => s
                .base_cls(py, base)
                .getattr(intern!(py, "items"))?
                .call1((source,))?,
            _ => source.call_method0(intern!(py, "items"))?,
        };
        if ordered {
            for key in s.dict_keys.bind(py).call1((source,))?.try_iter()? {
                if !self.s.plain_key(&key?) {
                    // Reading link order would run a key's own __hash__;
                    // keep insertion order instead and say so.
                    ordered = false;
                    self.count(py, intern!(py, "mapping_order"))?;
                    break;
                }
            }
        }
        let (mut limit, mut reason) = (self.max_items, intern!(py, "max_items"));
        if let Some(nodes) = self.max_nodes {
            if !names && limit.is_none_or(|limit| nodes < limit) {
                // Each entry not already memoized costs a node, so the node
                // budget also bounds the selection and its temporary memory.
                (limit, reason) = (Some(nodes), intern!(py, "max_nodes"));
            }
        }
        // A builtin view's length is read in C and bounds the selection, so
        // one allocation holds it.
        let size = view.len()? as u64;
        let size = limit.map_or(size, |limit| size.min(limit.saturating_add(1))) as usize;
        let mut chosen = Chosen {
            entries: Vec::with_capacity(size),
            slots: AddressMap::with_capacity_and_hasher(size, Default::default()),
            pulled: 0,
            cap: limit.map(|limit| limit.saturating_add(1)),
        };
        self.select(py, source, &view, ordered, preferred, &mut chosen)?;
        if ordered {
            // Link order gave keys only; read their values from the view.
            let mut remaining = chosen.entries.len();
            for item in view.try_iter()? {
                if remaining == 0 {
                    break;
                }
                let (key, value) = pair(s, item?)?;
                if let Some(&slot) = chosen.slots.get(&address(&key)) {
                    if chosen.entries[slot].1.is_none() {
                        chosen.entries[slot].1 = Some(value);
                        remaining -= 1;
                    }
                }
            }
        }
        self.own_count(py, source, &mut chosen)?;
        let target = captured.cast::<PyDict>().map_err(PyErr::from)?;
        let mut count = 0;
        for (key, item) in chosen.entries {
            let Some(item) = item else {
                continue;
            };
            if limit.is_some_and(|limit| count >= limit) {
                self.count(py, reason)?;
                break;
            }
            if !self.has_nodes() {
                self.count(py, intern!(py, "max_nodes"))?;
                break;
            }
            let converted_key = if names && is_exact_str(self, &key) {
                self.code_text(py, &key)?
            } else {
                self.value(py, &key, depth + 1)?
            };
            let converted_value = if depth != 0 {
                self.value(py, &item, depth + 1)?
            } else {
                self.root_value(py, &item)?
            };
            if let Err(err) = target.set_item(converted_key, converted_value) {
                if !err.is_instance_of::<PyTypeError>(py) {
                    return Err(Halt::Py(err));
                }
                self.count(py, intern!(py, "unsupported_key"))?;
            }
            count += 1;
        }
        Ok(())
    }

    /// The context's own instance dict, captured from inside its own call
    /// (through the context, or a reference to the dict): its `_nodes`
    /// holds the running count when the Python walk reads its entries.
    /// Only a selected key that is the interned `"_nodes"`, the very object
    /// attribute names are, while the count is not written, whose value is
    /// the object the context holds (read as an attribute), can make
    /// `source` that dict. The walker then writes the count to the context,
    /// which the Python walk holds there at every step, and reads the entry
    /// of `source` again by iterating its storage, which runs no code: only
    /// the context's own dict now holds the very int written. Neither step
    /// reads `ctx.__dict__`, which would make the dict exist where the
    /// Python walk never does (`rust/DESIGN.md`, "State reads").
    fn own_count<'py>(
        &mut self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        chosen: &mut Chosen<'py>,
    ) -> PyResult<()> {
        let name = intern!(py, "_nodes");
        if self.written == Some(self.nodes) {
            return Ok(());
        }
        let Some(slot) = chosen.entries.iter().position(|(key, _)| key.is(name)) else {
            return Ok(());
        };
        let Some(found) = &chosen.entries[slot].1 else {
            return Ok(());
        };
        let Ok(dict) = source.cast::<PyDict>() else {
            return Ok(());
        };
        let ctx = self.ctx(py).clone();
        if !found.is(&ctx.getattr(name)?) {
            return Ok(());
        }
        let count = self.nodes.into_pyobject(py)?.into_any();
        ctx.setattr(name, &count)?;
        self.written = Some(self.nodes);
        let now = dict.iter().find(|(key, _)| key.is(name));
        if now.is_some_and(|(_, value)| value.is(&count)) {
            chosen.entries[slot].1 = Some(count);
        }
        Ok(())
    }

    /// `ValuesMixin._attribute_mapping`: an attribute dict, whose keys are
    /// names, not string data. `private` is the bookkeeping attribute of a
    /// snapshot copy captured again; its caller restores it. Leaving it out
    /// builds a dict of the other entries, which hashes a key that is not a
    /// plain one with its own `__hash__`, and compares a later key with it,
    /// as the Python walk's dict comprehension does ([`State::insert`]).
    pub(crate) fn attribute_mapping<'py>(
        &mut self,
        py: Python<'py>,
        attrs: &Bound<'py, PyAny>,
        depth: i64,
        private: Option<&Bound<'py, PyAny>>,
    ) -> Walk<Bound<'py, PyAny>> {
        let attrs = match private {
            None => attrs.clone(),
            Some(private) => {
                let kept = PyDict::new(py);
                let mut mixed = false;
                for item in self.s.dict_items.bind(py).call1((attrs,))?.try_iter()? {
                    let (key, value) = pair(self.s, item?)?;
                    if is_exact_str(self, &key) && key.eq(private)? {
                        continue;
                    }
                    self.insert(py, &kept, &key, &value, &mut mixed)?;
                }
                kept.into_any()
            }
        };
        self.add_named(py, &id(&attrs))?;
        let dict = self.s.dict_cls.bind(py).clone();
        self.mapping(py, &attrs, depth, None, false, &dict)
    }

    /// `ValuesMixin._root_value`: a `RecursionError` while capturing `item`,
    /// raised by application code the walk runs, a subtree it runs in the
    /// Python walk, or CPython's own checks, forgets the copies the call
    /// added since and leaves a placeholder, as there.
    pub(crate) fn root_value<'py>(
        &mut self,
        py: Python<'py>,
        item: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        let mark = self.objects(py).len();
        let captured = self.value(py, item, 1);
        match captured {
            Err(Halt::Py(err)) if err.is_instance_of::<PyRecursionError>(py) => {
                // Forget the copies the overflow left half built.
                let objects = self.objects(py).clone();
                forget_newest(py, &objects, objects.len().saturating_sub(mark))?;
                let omitted = self.omit(py, intern!(py, "recursion"), item)?;
                self.remember(py, &id(item), item, omitted)
            }
            other => other,
        }
    }
}

/// Removes the `count` newest entries of `dict`.
fn forget_newest(py: Python<'_>, dict: &Bound<'_, PyDict>, count: usize) -> PyResult<()> {
    if count == 0 {
        return Ok(());
    }
    let newest_first = dict.call_method0(intern!(py, "__reversed__"))?;
    let keys: Vec<_> = newest_first
        .try_iter()?
        .take(count)
        .collect::<PyResult<_>>()?;
    for key in keys {
        dict.del_item(key)?;
    }
    Ok(())
}
