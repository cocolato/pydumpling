//! Key selection for `ValuesMixin._mapping`.
//!
//! Keys are selected before any entry is converted: preferred names first,
//! then the rest, each group in the mapping's own order, `limit + 1` keys in
//! all so the copy loop can report truncation. `__builtins__` is never
//! selected. Hashing or comparing a key runs only builtin code.
//!
//! [`pair`] unpacks `key, value = item` for the walk's readers, through the
//! item's own iterator where it is not an exact pair.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PySet, PyTuple};

use crate::address::AddressMap;
use crate::state::State;
use crate::statics::{address, type_address, Statics};
use crate::types::class_dicts;
use crate::walk::{is_exact_str, str_equals};
use crate::walker::Walk;

/// `key, value = item`. The builtin views the walk reads yield exact
/// pairs; anything else is unpacked as `UNPACK_SEQUENCE` does, through its
/// own iterator, with the interpreter's errors.
pub(crate) fn pair<'py>(
    s: &Statics,
    item: Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    if let Ok(items) = item.cast_exact::<PyTuple>() {
        if items.len() == 2 {
            return Ok((items.get_item(0)?, items.get_item(1)?));
        }
    }
    let mut iterator = match item.try_iter() {
        Ok(iterator) => iterator,
        Err(err) if err.is_instance_of::<PyTypeError>(item.py()) && !iterable(s, &item)? => {
            let name = item.get_type().name()?;
            return Err(PyTypeError::new_err(format!(
                "cannot unpack non-iterable {name} object"
            )));
        }
        Err(err) => return Err(err),
    };
    let mut taken = Vec::with_capacity(2);
    while taken.len() < 2 {
        match iterator.next().transpose()? {
            Some(next) => taken.push(next),
            None => {
                return Err(PyValueError::new_err(format!(
                    "not enough values to unpack (expected 2, got {})",
                    taken.len()
                )))
            }
        }
    }
    if iterator.next().transpose()?.is_some() {
        // From 3.14 the interpreter also names the length of an exact list,
        // tuple or dict.
        let sized = [s.list, s.tuple, s.dict].contains(&type_address(&item))
            && item.py().version_info() >= (3, 14);
        let message = if sized {
            format!(
                "too many values to unpack (expected 2, got {})",
                item.len()?
            )
        } else {
            "too many values to unpack (expected 2)".to_owned()
        };
        return Err(PyValueError::new_err(message));
    }
    let value = taken.pop();
    match (taken.pop(), value) {
        (Some(key), Some(value)) => Ok((key, value)),
        _ => unreachable!("two values were taken"),
    }
}

/// Whether the type of `item` has an iterator slot, or is a sequence and
/// not a dict: whether a class along its MRO defines `__iter__`, or
/// `__getitem__` outside a dict. For anything else the interpreter words
/// the error of unpacking it itself.
fn iterable(s: &Statics, item: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = item.py();
    let (mut getitem, mut mapping) = (false, false);
    for (cls, namespace) in class_dicts(s, item.get_type().as_any())? {
        mapping |= address(&cls) == s.dict;
        if namespace.contains(intern!(py, "__iter__"))? {
            return Ok(true);
        }
        getitem |= namespace.contains(intern!(py, "__getitem__"))?;
    }
    Ok(getitem && !mapping)
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
    fn next(
        &mut self,
        s: &Statics,
    ) -> Walk<Option<(Bound<'py, PyAny>, Option<Bound<'py, PyAny>>)>> {
        match self {
            KeyOrder::Linked(keys) => Ok(keys.next().transpose()?.map(|key| (key, None))),
            KeyOrder::Items(items) => match items.next().transpose()? {
                Some(item) => Ok(pair(s, item).map(|(key, value)| Some((key, Some(value))))?),
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
        preferred: Option<&Bound<'py, PySet>>,
        chosen: &mut Chosen<'py>,
    ) -> Walk<()> {
        let builtins = intern!(py, "__builtins__");
        let is_builtins = |key: &Bound<'py, PyAny>| -> Walk<bool> {
            Ok(is_exact_str(self, key) && str_equals(key, builtins)?)
        };
        let is_preferred = |key: &Bound<'py, PyAny>| -> Walk<bool> {
            Ok(match preferred {
                Some(preferred) => is_exact_str(self, key) && preferred.contains(key)?,
                None => false,
            })
        };
        let any_preferred = preferred.is_some();
        if any_preferred {
            let mut keys = KeyOrder::new(self, py, source, view, ordered)?;
            while !chosen.full() {
                let Some((key, value)) = keys.next(self.s)? else {
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
            let Some((key, value)) = keys.next(self.s)? else {
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
