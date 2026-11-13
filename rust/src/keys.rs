//! Key selection for `ValuesMixin._mapping`.
//!
//! Keys are selected before any entry is converted: preferred names first,
//! then the rest, each group in the mapping's own order, `limit + 1` keys in
//! all so the copy loop can report truncation. `__builtins__` is never
//! selected. Hashing or comparing a key runs only builtin code.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PySet, PyTuple};

use crate::address::AddressMap;
use crate::statics::address;
use crate::walk::{is_exact_str, str_equals};
use crate::walker::{Halt, State, Walk};

/// Unpacks `key, value = item` for the exact pairs builtin views yield;
/// unpacking anything else is the Python walk's to do.
pub(crate) fn pair<'py>(item: Bound<'py, PyAny>) -> Walk<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    match item.cast_exact::<PyTuple>() {
        Ok(items) if items.len() == 2 => Ok((items.get_item(0)?, items.get_item(1)?)),
        _ => Err(Halt::Bail),
    }
}

/// A mapping's keys in its own order, each with its value when the order
/// comes from the items view itself (`_key_order`).
enum KeyOrder<'py> {
    /// `OrderedDict.__iter__`: link order, keys only.
    Linked(Bound<'py, PyIterator>),
    /// The items view, in insertion order.
    Items(Bound<'py, PyIterator>),
}

impl<'py> KeyOrder<'py> {
    fn new(
        state: &State,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        view: &Bound<'py, PyAny>,
        ordered: bool,
    ) -> PyResult<Self> {
        Ok(if ordered {
            KeyOrder::Linked(state.s.ordered_iter.bind(py).call1((source,))?.try_iter()?)
        } else {
            KeyOrder::Items(view.try_iter()?)
        })
    }

    #[allow(clippy::type_complexity)]
    fn next(&mut self) -> Walk<Option<(Bound<'py, PyAny>, Option<Bound<'py, PyAny>>)>> {
        match self {
            KeyOrder::Linked(keys) => Ok(keys.next().transpose()?.map(|key| (key, None))),
            KeyOrder::Items(items) => match items.next().transpose()? {
                Some(item) => pair(item).map(|(key, value)| Some((key, Some(value)))),
                None => Ok(None),
            },
        }
    }
}

/// The selected keys, deduplicated by identity like `dict.fromkeys(ids)`.
pub(crate) struct Chosen<'py> {
    pub entries: Vec<(Bound<'py, PyAny>, Option<Bound<'py, PyAny>>)>,
    pub slots: AddressMap<usize>,
    /// Keys pulled from the selection, duplicates included (`islice`).
    pub pulled: u64,
    pub cap: Option<u64>,
}

impl<'py> Chosen<'py> {
    fn full(&self) -> bool {
        self.cap.is_some_and(|cap| self.pulled >= cap)
    }

    fn push(&mut self, key: Bound<'py, PyAny>, value: Option<Bound<'py, PyAny>>) {
        self.pulled += 1;
        let next = self.entries.len();
        if let std::collections::hash_map::Entry::Vacant(slot) = self.slots.entry(address(&key)) {
            slot.insert(next);
            crate::address::push(&mut self.entries, (key, value));
        }
    }
}

impl State {
    /// Selects up to `chosen.cap` keys: preferred names first, then the
    /// rest, each group in the mapping's own order, never `__builtins__`.
    pub(crate) fn select<'py>(
        &self,
        py: Python<'py>,
        source: &Bound<'py, PyAny>,
        view: &Bound<'py, PyAny>,
        ordered: bool,
        preferred: &Bound<'py, PySet>,
        chosen: &mut Chosen<'py>,
    ) -> Walk<()> {
        let builtins = intern!(py, "__builtins__");
        let is_builtins = |key: &Bound<'py, PyAny>| -> Walk<bool> {
            Ok(is_exact_str(self, key) && str_equals(key, builtins)?)
        };
        let is_preferred = |key: &Bound<'py, PyAny>| -> Walk<bool> {
            Ok(is_exact_str(self, key) && preferred.contains(key)?)
        };
        let any_preferred = !preferred.is_empty();
        if any_preferred {
            let mut keys = KeyOrder::new(self, py, source, view, ordered)?;
            while !chosen.full() {
                let Some((key, value)) = keys.next()? else {
                    break;
                };
                if is_preferred(&key)? && !is_builtins(&key)? {
                    chosen.push(key, value);
                }
            }
        }
        if chosen.full() {
            return Ok(());
        }
        let mut keys = KeyOrder::new(self, py, source, view, ordered)?;
        while !chosen.full() {
            let Some((key, value)) = keys.next()? else {
                break;
            };
            if is_builtins(&key)? || (any_preferred && is_preferred(&key)?) {
                continue;
            }
            chosen.push(key, value);
        }
        Ok(())
    }
}
