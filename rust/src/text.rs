//! Budgets, counters, placeholders, and bounded text: the small helpers of
//! `SnapshotContext`, each with a fast path for the common case and the
//! Python helper itself for the rest, so every side effect stays the same.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::statics::{address, type_address, Statics};
use crate::types::fresh_dict;
use crate::walker::{Entry, Halt, Slot, State, Walk};

/// Whether `chars` code points surely fit in `limit` UTF-8 bytes: each takes
/// at most four, so the Python helper's encode-and-compare returns the
/// value itself.
fn surely_fits(chars: usize, limit: u64) -> bool {
    (chars as u128) * 4 <= u128::from(limit)
}

/// Whether the exact str `value` fits in `limit` UTF-8 bytes, so that
/// `SnapshotContext._bounded_text` returns it and counts nothing: one that
/// surely fits, or an ASCII one, a byte per code point, no longer than the
/// limit. `str.isascii` is C.
fn fits(py: Python<'_>, value: &Bound<'_, PyAny>, limit: u64) -> PyResult<bool> {
    let chars = value.len()?;
    if surely_fits(chars, limit) {
        return Ok(true);
    }
    Ok(chars as u64 <= limit && value.call_method0(intern!(py, "isascii"))?.is_truthy()?)
}

/// Reads a budget as `DumpOptions` validates it: `Some(None)` for `None`,
/// `Some(Some(limit))` for a positive exact int, saturated to `u64`, and
/// `None` for anything the Python walk alone compares as it would.
pub(crate) fn budget(s: &Statics, value: &Bound<'_, PyAny>) -> Option<Option<u64>> {
    if value.is_none() {
        return Some(None);
    }
    if type_address(value) != s.int {
        return None;
    }
    match value.extract::<u64>() {
        Ok(0) => None,
        Ok(limit) => Some(Some(limit)),
        // Too big for u64, or negative.
        Err(_) => match value.gt(0) {
            Ok(true) => Some(Some(u64::MAX)),
            _ => None,
        },
    }
}

impl State {
    /// `SnapshotContext._has_nodes`.
    pub(crate) fn has_nodes(&self) -> bool {
        self.max_nodes.is_none_or(|limit| self.nodes < limit)
    }

    /// Whether `depth` reached `max_depth`.
    pub(crate) fn depth_reached(&self, depth: i64) -> bool {
        self.max_depth
            .is_some_and(|limit| i128::from(depth) >= i128::from(limit))
    }

    /// `self._truncation[reason]`, which counting adds one to. An exact
    /// int is read without running code; another tally the Python walk
    /// adds to with its own `__add__`, so the call goes to it.
    fn tally(&self, py: Python<'_>, reason: &Bound<'_, PyString>) -> Walk<i64> {
        match self.truncation.bind(py).get_item(reason)? {
            None => Ok(0),
            Some(count) if type_address(&count) == self.s.int => match count.extract::<i64>() {
                Ok(count) if count < i64::MAX => Ok(count),
                _ => Err(Halt::Bail),
            },
            Some(_) => Err(Halt::Bail),
        }
    }

    /// `self._truncation[reason] += 1`.
    pub(crate) fn count(&self, py: Python<'_>, reason: &Bound<'_, PyString>) -> Walk<()> {
        let current = self.tally(py, reason)?;
        self.truncation.bind(py).set_item(reason, current + 1)?;
        Ok(())
    }

    /// Hands the call back before a Python helper that may count `reason`
    /// runs Python code: a hook or the tally's own `__add__` must run in
    /// the Python walk alone, not once here and again after a hand-back.
    fn before_helper(&self, py: Python<'_>, reason: &Bound<'_, PyString>) -> Walk<()> {
        self.tally(py, reason)?;
        self.before_python(py)
    }

    /// `SnapshotContext._remember`.
    pub(crate) fn remember<'py>(
        &mut self,
        original: &Bound<'py, PyAny>,
        captured: Bound<'py, PyAny>,
    ) -> Bound<'py, PyAny> {
        let key = address(original);
        if type_address(&captured) == self.s.truncated_type {
            self.truncated.push(key);
        }
        let entry = Entry {
            slot: Slot::Value(captured.clone().unbind()),
            original: original.clone().unbind(),
        };
        if self
            .memo
            .insert(key, entry)
            .is_some_and(|slot| slot < self.mirrored)
        {
            // `ctx.objects` still holds the copy this one replaces.
            self.dirty.push(key);
        }
        captured
    }

    /// `SnapshotContext._text` for an exact str.
    pub(crate) fn text<'py>(
        &self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        match self.max_string_bytes {
            Some(limit) if !fits(py, value, limit)? => {
                self.before_helper(py, intern!(py, "max_string_bytes"))?;
                Ok(self.ctx(py)?.call_method1(intern!(py, "_text"), (value,))?)
            }
            _ => Ok(value.clone()),
        }
    }

    /// `SnapshotContext._code_text` for an exact str.
    pub(crate) fn code_text<'py>(
        &self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        match self.code_text_limit {
            Some(limit) if !fits(py, value, limit)? => {
                self.before_helper(py, intern!(py, "code_metadata"))?;
                Ok(self
                    .ctx(py)?
                    .call_method1(intern!(py, "_code_text"), (value,))?)
            }
            _ => Ok(value.clone()),
        }
    }

    /// `SnapshotContext._bounded_bytes` with the string budget.
    ///
    /// Slicing nonempty exact bytes to its full length returns the object
    /// itself; an empty one may become the shared empty bytes.
    pub(crate) fn bounded_bytes<'py>(
        &self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        if type_address(value) == self.s.bytes
            && value.len().is_ok_and(|size| {
                size > 0
                    && self
                        .max_string_bytes
                        .is_none_or(|limit| size as u64 <= limit)
            })
        {
            return Ok(value.clone());
        }
        let reason = intern!(py, "max_string_bytes");
        self.before_helper(py, reason)?;
        let args = (value, self.string_limit.bind(py), reason);
        Ok(self
            .ctx(py)?
            .call_method1(intern!(py, "_bounded_bytes"), args)?)
    }

    /// `SnapshotContext._type_name`: read through the type descriptors.
    pub(crate) fn type_name<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        let cls = value.get_type();
        // A class's `__module__` is looked up in its dict.
        if self.info(py, &cls)?.hostile {
            return Err(Halt::Bail);
        }
        let meta = cls.get_type();
        let module = self.s.class_module.bind(py).call1((&cls, &meta))?;
        let name = self.s.class_qualname.bind(py).call1((&cls, &meta))?;
        if type_address(&name) != self.s.str_ {
            // A str subclass name runs its own `__radd__` and slicing.
            return Err(Halt::Bail);
        }
        let module = if type_address(&module) == self.s.str_ {
            module
        } else {
            intern!(py, "?").clone().into_any()
        };
        let text = if module.eq(intern!(py, "builtins"))? {
            name
        } else {
            module.add(intern!(py, "."))?.add(&name)?
        };
        self.code_text(py, &text)
    }

    /// A `TruncatedValue(reason, type_name)`, built as its `__init__` would,
    /// through the instance-dict descriptor the class had at import: no
    /// hook application code gave the class runs.
    pub(crate) fn truncated<'py>(
        &self,
        py: Python<'py>,
        reason: &Bound<'py, PyAny>,
        type_name: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        let s = self.s;
        let placeholder = s.object_new.bind(py).call1((s.truncated.bind(py),))?;
        let attributes = fresh_dict(&s.truncated_dict, &placeholder)?;
        attributes.set_item(intern!(py, "reason"), reason)?;
        attributes.set_item(intern!(py, "type_name"), type_name)?;
        Ok(placeholder)
    }

    /// `SnapshotContext._omit(reason, value)`.
    pub(crate) fn omit<'py>(
        &mut self,
        py: Python<'py>,
        reason: &Bound<'py, PyString>,
        value: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        self.count(py, reason)?;
        let type_name = if value.is_none() {
            intern!(py, "object").clone().into_any()
        } else {
            self.type_name(py, value)?
        };
        self.truncated(py, reason.as_any(), &type_name)
    }

    /// `"<%s snapshot>" % type_name`, for the exact str [`State::type_name`]
    /// returns.
    pub(crate) fn snapshot_repr<'py>(
        &self,
        py: Python<'py>,
        type_name: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        if type_address(type_name) == self.s.str_ {
            if let Ok(text) = type_name.cast::<PyString>().map_err(PyErr::from)?.to_str() {
                return Ok(PyString::new(py, &format!("<{text} snapshot>")).into_any());
            }
        }
        // A lone surrogate does not convert; Python formats it.
        Ok(PyString::new(py, "<%s snapshot>").rem(type_name)?)
    }

    /// The placeholder of an exact tuple or frozenset met again while it is
    /// still being filled, built on first use and then memoized.
    pub(crate) fn placeholder<'py>(
        &mut self,
        py: Python<'py>,
        key: usize,
    ) -> Walk<Bound<'py, PyAny>> {
        let Some(entry) = self.memo.get(key) else {
            unreachable!("a placeholder is only built for a memoized original");
        };
        let original = entry.original.bind(py).clone();
        let type_name = self.type_name(py, &original)?;
        let reason = self.s.recursive_immutable.bind(py).clone();
        let placeholder = self.truncated(py, &reason, &type_name)?;
        Ok(self.remember(&original, placeholder))
    }
}

/// `options.max_items`, `max_nodes`, `max_depth` and `max_string_bytes`,
/// for `options` an exact `DumpOptions`, if reading each runs no code, as
/// the walker reads them all where the Python walk reads only those it
/// needs: from an instance dict of exact str keys (a dict subclass assigned
/// as `__dict__` too, which attribute lookup reads as a dict), while
/// `DumpOptions` holds no hook, and for each, if anything, `None` or an
/// int, a default the instance dict overrides, as the dataclass does.
pub(crate) fn plain_budgets<'py>(
    py: Python<'py>,
    s: &Statics,
    options: &Bound<'py, PyAny>,
) -> PyResult<Option<[Bound<'py, PyAny>; 4]>> {
    let names = [
        intern!(py, "max_items"),
        intern!(py, "max_nodes"),
        intern!(py, "max_depth"),
        intern!(py, "max_string_bytes"),
    ];
    let Some(defaults) = s.options_own.entries(py, s, names)? else {
        return Ok(None);
    };
    let int_or_none = |entry: &Bound<'_, PyAny>| entry.is_none() || type_address(entry) == s.int;
    if !defaults.iter().flatten().all(int_or_none) {
        return Ok(None);
    }
    let Ok(fields) = s
        .options_dict
        .bind(py)
        .call1((options,))?
        .cast_into::<PyDict>()
    else {
        return Ok(None);
    };
    if !s.str_keys(&fields) {
        return Ok(None);
    }
    let [items, nodes, depth, string] = names.map(|name| fields.get_item(name));
    Ok(match (items?, nodes?, depth?, string?) {
        (Some(items), Some(nodes), Some(depth), Some(string)) => {
            Some([items, nodes, depth, string])
        }
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::surely_fits;

    #[test]
    fn four_bytes_per_code_point_is_the_bound() {
        assert!(surely_fits(1024, 4096));
        assert!(!surely_fits(1025, 4096));
        assert!(surely_fits(0, 0));
        assert!(surely_fits(usize::MAX / 8, u64::MAX));
    }
}
