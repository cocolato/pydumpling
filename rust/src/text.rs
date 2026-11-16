//! Budgets, counters, placeholders, and bounded text: the helpers of
//! `SnapshotContext` the walk uses, done natively with the operations the
//! Python helpers make, in their order, through the C API.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyInt, PySlice, PyString};

use crate::state::State;
use crate::statics::{type_address, Statics};
use crate::types::{fresh_dict, TypeInfo};
use crate::walker::Walk;

/// Whether `chars` code points surely fit in `limit` UTF-8 bytes: each takes
/// at most four, so `_bounded_text`'s encode-and-compare keeps the value.
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

/// `[:limit]`, for a limit that fits, as slicing reads it.
fn head(py: Python<'_>, limit: u64) -> Bound<'_, PySlice> {
    PySlice::new(py, 0, isize::try_from(limit).unwrap_or(isize::MAX), 1)
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

    /// `self._truncation[reason] += 1`: the subscript, the in-place add and
    /// the store the Python walk makes. An exact `Counter` is read in the
    /// dict, where a missing tally is 0 without its `__missing__`, and an
    /// exact int tally is added to natively; anything else runs its own
    /// code, as in the Python walk.
    pub(crate) fn count(&mut self, py: Python<'_>, reason: &Bound<'_, PyString>) -> Walk<()> {
        let (truncation, counter, plain) = self.truncation(py)?;
        self.started = true;
        if !counter {
            let iadd = self.s.iadd.bind(py);
            return self.application(py, || {
                let tally = truncation.get_item(reason)?;
                truncation.set_item(reason, iadd.call1((tally, 1))?)
            });
        }
        let tallies = truncation.cast_into::<PyDict>().map_err(PyErr::from)?;
        let (iadd, int) = (self.s.iadd.bind(py), self.s.int);
        if !plain {
            // A key's own `__eq__` may run in either lookup.
            return self.application(py, || {
                let tally = match tallies.get_item(reason)? {
                    None => PyInt::new(py, 1).into_any(),
                    Some(tally) if type_address(&tally) == int => tally.add(1)?,
                    Some(tally) => iadd.call1((tally, 1))?,
                };
                tallies.set_item(reason, tally)
            });
        }
        let tally = match tallies.get_item(reason)? {
            None => PyInt::new(py, 1).into_any(),
            Some(tally) if type_address(&tally) == int => match tally.extract::<i64>() {
                Ok(count) if count < i64::MAX => PyInt::new(py, count + 1).into_any(),
                _ => tally.add(1)?,
            },
            Some(tally) => {
                // Its own `__iadd__` may run, and change what the store
                // below compares.
                return self.application(py, || tallies.set_item(reason, iadd.call1((tally, 1))?));
            }
        };
        tallies.set_item(reason, tally)?;
        Ok(())
    }

    /// `SnapshotContext._remember`, for the original whose id is `id`.
    pub(crate) fn remember<'py>(
        &mut self,
        py: Python<'py>,
        id: &Bound<'py, PyInt>,
        original: &Bound<'py, PyAny>,
        captured: Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        self.started = true;
        self.objects(py).set_item(id, &captured)?;
        // Under an int of its own, as the Python walk's second `id()` makes
        // one: a walk that captures both dicts copies each key as a value.
        let (sources, key) = (self.sources(py)?, crate::statics::id(original));
        if type_address(&sources) == self.s.dict {
            sources.set_item(key, original)?;
        } else {
            // Its own `__setitem__` may run.
            self.application(py, || sources.set_item(key, original))?;
        }
        Ok(captured)
    }

    /// `SnapshotContext._text` for an exact str.
    pub(crate) fn text<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        let limit = self.max_string_bytes;
        self.bounded_text(py, value, limit, intern!(py, "max_string_bytes"))
    }

    /// `SnapshotContext._code_text`. A `_code_text_limit` other than
    /// `None` or a positive exact int, which only Python code rebinding it
    /// makes, bounds the text with its own code, as in the Python walk.
    pub(crate) fn code_text<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        let reason = intern!(py, "code_metadata");
        let limit = self.code_text_limit(py)?;
        match budget(self.s, &limit) {
            Some(read) => self.bounded_text(py, value, read, reason),
            None => self.bounded_object(py, value, &limit, reason),
        }
    }

    /// `SnapshotContext._bounded_text(value, limit, reason)`: the str
    /// slice, `encode("utf-8", "surrogatepass")`, bytes slice and
    /// `decode("utf-8", "ignore")` it makes (with `context._utf8_prefix`),
    /// on an exact str through the C API, whose codecs and error handlers
    /// are C. Any other value, which only a type name made by a str
    /// subclass's own code is, runs the same operations generically.
    fn bounded_text<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
        limit: Option<u64>,
        reason: &Bound<'py, PyString>,
    ) -> Walk<Bound<'py, PyAny>> {
        let Some(limit) = limit else {
            return Ok(value.clone());
        };
        if type_address(value) != self.s.str_ {
            let limit = PyInt::new(py, limit).into_any();
            return self.bounded_object(py, value, &limit, reason);
        }
        if fits(py, value, limit)? {
            return Ok(value.clone());
        }
        let encode = (intern!(py, "utf-8"), intern!(py, "surrogatepass"));
        let prefix = value.get_item(head(py, limit))?;
        let encoded = if value.len()? == prefix.len()? {
            let encoded = prefix.call_method1(intern!(py, "encode"), encode)?;
            if encoded.len()? as u64 <= limit {
                return Ok(value.clone());
            }
            encoded
        } else {
            prefix.call_method1(intern!(py, "encode"), encode)?
        };
        self.count(py, reason)?;
        // `_utf8_prefix(value, limit)`: `value[:limit]` is `prefix`, and
        // encoding it again gives the same bytes.
        let encoded = encoded.cast_into::<PyBytes>().map_err(PyErr::from)?;
        let bytes = encoded.as_bytes();
        let cut = PyBytes::new(py, &bytes[..bytes.len().min(limit as usize)]);
        let decoded = PyString::from_encoded_object(cut.as_any(), Some(c"utf-8"), Some(c"ignore"))?;
        Ok(decoded.into_any())
    }

    /// `_bounded_text` on a value other than an exact str, or with a limit
    /// other than a positive int, whose own code may run at every step, as
    /// in the Python walk.
    fn bounded_object<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
        limit: &Bound<'py, PyAny>,
        reason: &Bound<'py, PyString>,
    ) -> Walk<Bound<'py, PyAny>> {
        if limit.is_none() {
            return Ok(value.clone());
        }
        // `value[:limit]` slices with `slice(None, limit, None)`.
        let stop = self.s.slice_cls.bind(py).call1((py.None(), limit))?;
        let encode = (intern!(py, "utf-8"), intern!(py, "surrogatepass"));
        let keep = self.application(py, || {
            let prefix = value.get_item(&stop)?;
            if value.len()? != prefix.len()? {
                return Ok(false);
            }
            let size = prefix.call_method1(intern!(py, "encode"), encode)?.len()?;
            size.into_pyobject(py)?.le(limit)
        })?;
        if keep {
            return Ok(value.clone());
        }
        self.count(py, reason)?;
        let decode = (intern!(py, "utf-8"), intern!(py, "ignore"));
        self.application(py, || {
            let encoded = value
                .get_item(&stop)?
                .call_method1(intern!(py, "encode"), encode)?;
            encoded
                .get_item(&stop)?
                .call_method1(intern!(py, "decode"), decode)
        })
    }

    /// `SnapshotContext._bounded_bytes(value, max_string_bytes,
    /// "max_string_bytes")` for exact bytes or a bytearray.
    ///
    /// Slicing nonempty exact bytes to its full length returns the object
    /// itself; an empty one may become the shared empty bytes.
    pub(crate) fn bounded_bytes<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        let size = value.len()? as u64;
        let limit = self.max_string_bytes;
        if type_address(value) == self.s.bytes && size > 0 && limit.is_none_or(|l| size <= l) {
            return Ok(value.clone());
        }
        if limit.is_some_and(|limit| size > limit) {
            self.count(py, intern!(py, "max_string_bytes"))?;
        }
        let slice = match limit {
            Some(limit) => head(py, limit),
            None => PySlice::full(py),
        };
        Ok(value.get_item(slice)?)
    }

    /// `SnapshotContext._type_name`: the class's `__module__` and
    /// `__qualname__`, read through the `type` descriptors, joined with the
    /// `+` the Python walk makes, then bounded as code text.
    pub(crate) fn type_name<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        let s = self.s;
        let cls = value.get_type();
        let meta = cls.get_type();
        let module_of = s.class_module.bind(py);
        // A class's `__module__` is looked up in its dict, where a key that
        // is not an exact str may run its `__eq__`; the dicts of the
        // builtin types the walk meets most hold only str keys.
        let module = if TypeInfo::exact_kind(s, type_address(value)).is_none()
            && self.info(py, &cls)?.hostile
        {
            self.application(py, || module_of.call1((&cls, &meta)))?
        } else {
            module_of.call1((&cls, &meta))?
        };
        let name = s.class_qualname.bind(py).call1((&cls, &meta))?;
        let module = if type_address(&module) == s.str_ {
            module
        } else {
            intern!(py, "?").clone().into_any()
        };
        // An exact str compared with an exact str runs no code.
        let text = if module.eq(intern!(py, "builtins"))? {
            name
        } else {
            let dotted = module.add(intern!(py, "."))?;
            if type_address(&name) == s.str_ {
                dotted.add(&name)?
            } else {
                // A str subclass's own `__radd__` runs.
                self.application(py, || dotted.add(&name))?
            }
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

    /// `"<%s snapshot>" % type_name`. Formatting anything but an exact str
    /// runs its own `__str__`, or reads it as a tuple of arguments.
    pub(crate) fn snapshot_repr<'py>(
        &mut self,
        py: Python<'py>,
        type_name: &Bound<'py, PyAny>,
    ) -> Walk<Bound<'py, PyAny>> {
        let template = intern!(py, "<%s snapshot>");
        if type_address(type_name) != self.s.str_ {
            return self.application(py, || template.rem(type_name));
        }
        if let Ok(text) = type_name.cast::<PyString>().map_err(PyErr::from)?.to_str() {
            return Ok(PyString::new(py, &format!("<{text} snapshot>")).into_any());
        }
        // A lone surrogate does not convert; Python formats it.
        Ok(template.rem(type_name)?)
    }
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
