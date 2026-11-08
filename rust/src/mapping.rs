//! `ValuesMixin._mapping` and `_attribute_mapping`: bounded mapping copies.
//!
//! Values are read through the builtin base's `items`, so no subclass
//! override or key hook runs; [`crate::keys`] selects which entries to copy.
//! Any other mapping's own `items` is application code, so the Python walk
//! reads it; only a frame's `f_locals` proxy is read here.

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet};

use crate::keys::{pair, Chosen};
use crate::memo::AddressMap;
use crate::statics::{address, type_address};
use crate::types::{Base, Kind};
use crate::walk::is_exact_str;
use crate::walker::{Halt, Slot, State, Walk};

impl State {
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
        self.enter(py)?;
        let captured = self.mapping_body(py, source, depth, priority, counted, container);
        self.leave();
        captured
    }

    fn mapping_body<'py>(
        &mut self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        depth: i64,
        priority: Option<&Bound<'py, PyAny>>,
        counted: bool,
        container: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        let s = self.s;
        let key = address(source);
        let cached = self.memo.get(key).map(|entry| match &entry.slot {
            Slot::Recursive => None,
            Slot::Value(cached) => Some(cached.bind(py).clone()),
        });
        if let Some(cached) = cached {
            // A mapping root may fit even when a deeper reference was omitted.
            let mut reusable = false;
            if let Some(cached) = cached {
                let cached_type = type_address(&cached);
                if cached_type == s.dict || cached_type == s.fake_dict_type {
                    return Ok(cached);
                }
                if cached_type == s.truncated_type {
                    // Reading an application's placeholder may run its own
                    // code: only an exact one with an exact dict of plain
                    // keys is read here.
                    match self.attributes(py, &cached)? {
                        Some(attrs) if s.plain_dict(&attrs) => {}
                        _ => return Err(Halt::Bail),
                    }
                    let reason = cached.getattr(intern!(py, "reason"))?;
                    if !is_exact_str(self, &reason) {
                        // An application's reason runs its own `__ne__`.
                        return Err(Halt::Bail);
                    }
                    reusable = !reason.ne(intern!(py, "max_depth"))?;
                } else if !s.plain_keys.contains(&cached_type)
                    && ![s.bytearray, s.list, s.tuple, s.set, s.frozenset].contains(&cached_type)
                {
                    // The Python walk's `isinstance` reads any other entry's
                    // `__class__`, which its class may answer with its own
                    // code, and a placeholder's `reason`: it runs them once.
                    return Err(Halt::Bail);
                }
            }
            if !reusable {
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
            let empty = container.call0()?;
            return Ok(self.remember(source, empty));
        }
        let empty = container.call0()?;
        let captured = self.remember(source, empty);
        // Frame variable names are identifiers, not application string
        // values, so a string budget must not rewrite them.
        let names = self.namespaces.bind(py).contains(key)?;
        let preferred = PySet::empty(py)?;
        if let Some(priority) = priority {
            for name in priority.try_iter()? {
                preferred.add(name?)?;
            }
        }
        match self.fill_mapping(py, source, depth, &captured, names, &preferred) {
            Ok(()) => {}
            Err(Halt::Py(err)) if err.is_instance_of::<PyRuntimeError>(py) => {
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
        preferred: &Bound<'py, PySet>,
    ) -> Walk<()> {
        let s = self.s;
        let info = self.info(py, &source.get_type())?;
        let (kind, mut ordered) = (info.kind, info.ordered);
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
            _ if type_address(source) == s.frame_locals => {
                source.call_method0(intern!(py, "items"))?
            }
            _ => return Err(Halt::Bail),
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
                let (key, value) = pair(item?)?;
                if let Some(&slot) = chosen.slots.get(&address(&key)) {
                    if chosen.entries[slot].1.is_none() {
                        chosen.entries[slot].1 = Some(value);
                        remaining -= 1;
                    }
                }
            }
        }
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
            if !self.s.plain_key(&converted_key) {
                // A copy's own `__hash__`, `FakeContainer`'s, is Python.
                self.before_python(py)?;
            }
            if let Err(err) = target.set_item(converted_key, converted_value) {
                match Halt::from(err) {
                    Halt::Py(err) if err.is_instance_of::<PyTypeError>(py) => {
                        self.count(py, intern!(py, "unsupported_key"))?;
                    }
                    halt => return Err(halt),
                }
            }
            count += 1;
        }
        Ok(())
    }

    /// `ValuesMixin._attribute_mapping`: an attribute dict, whose keys are
    /// names, not string data. `private` is the bookkeeping attribute of a
    /// snapshot copy captured again; its caller restores it.
    pub(crate) fn attribute_mapping<'py>(
        &mut self,
        py: Python<'py>,
        attrs: &Bound<'py, PyAny>,
        depth: i64,
        private: Option<&Bound<'py, PyAny>>,
    ) -> Walk<Bound<'py, PyAny>> {
        self.enter(py)?;
        let captured = self.attribute_mapping_body(py, attrs, depth, private);
        self.leave();
        captured
    }

    fn attribute_mapping_body<'py>(
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
                for item in self.s.dict_items.bind(py).call1((attrs,))?.try_iter()? {
                    let (key, value) = pair(item?)?;
                    if !self.s.plain_key(&key) {
                        // Building the filtered dict hashes every key.
                        return Err(Halt::Bail);
                    }
                    if !is_exact_str(self, &key) || key.ne(private)? {
                        kept.set_item(key, value)?;
                    }
                }
                kept.into_any()
            }
        };
        let id = address(&attrs);
        let namespaces = self.namespaces.bind(py);
        if !namespaces.contains(id)? {
            namespaces.add(id)?;
            self.added_namespaces.push(id);
        }
        let dict = self.s.dict_cls.bind(py).clone();
        self.mapping(py, &attrs, depth, None, false, &dict)
    }

    /// `ValuesMixin._root_value`. The native walk never overflows: where
    /// the Python walk could, the whole call already went back to Python.
    fn root_value<'py>(
        &mut self,
        py: Python<'py>,
        item: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        self.enter(py)?;
        let captured = self.value(py, item, 1);
        self.leave();
        captured
    }
}
