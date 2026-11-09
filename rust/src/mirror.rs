//! Keeps the walker's memo and the context's `objects` in step.
//!
//! The Python walk memoizes every copy in `ctx.objects`, keyed by the
//! original's id, and keeps the original alive in `ctx._sources`. The
//! walker memoizes in [`crate::memo::Memo`] and writes those dicts lazily:
//! the oldest [`State::mirrored`] entries are in `ctx.objects`, in the same
//! order, and [`crate::flush`] writes the rest only before Python code reads
//! them, on a hand-back or a copy of the context. Python adds copies to
//! `ctx.objects` by appending, so afterwards the walker adopts just the new
//! tail; one that replaces or empties the dict resets the memo, as it
//! resets the Python walk's. Entries Python removes, the memo lets go of,
//! as the Python walk then holds them nowhere; at every sync the walker
//! checks that the last entries it saw are still where it saw them, as
//! removing some while adding others shows in no count. When Python code
//! changes `objects`, removes entries or replaces a dict while the walker
//! holds unwritten copies, the walker writes them first, after what Python
//! added. Additions to `_sources` alone, as `exception()` makes, need no
//! write. The walker looks originals up in `_sources` only while every key
//! it read there is of a plain kind, whose lookup runs no code; a key a
//! removal it does not notice lets in, it never reads.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt};

use crate::shadow::attributes;
use crate::statics::type_address;
use crate::walker::{Entry, Slot, State};

impl State {
    /// Brings the memo in line with `objects`, which Python code may have
    /// added to, emptied, or replaced since the last sync. `overwrites`
    /// also adopts the depth placeholders the Python walk replaced in place.
    pub(crate) fn sync(
        &mut self,
        py: Python<'_>,
        objects: Bound<'_, PyDict>,
        sources: Bound<'_, PyDict>,
        overwrites: bool,
    ) -> PyResult<()> {
        let same = match &self.mirror {
            Some((known, kept)) => known.bind(py).is(&objects) && kept.bind(py).is(&sources),
            None => false,
        };
        // Python code removed entries: a dict with fewer than the walker
        // saw has none where it saw the last. Removals alongside additions,
        // as another Python walk makes during a hand-back, show in no count,
        // but move that entry; only a removal that leaves it where it was
        // goes unnoticed.
        let moved = same && !self.follows(py, &objects, &sources)?;
        let flushes = self.unwritten() && (!same || moved || objects.len() != self.mirrored);
        if flushes {
            // The walker's copies go into the dicts it mirrored, after what
            // Python added; the flush then takes the memo from `objects`.
            self.flush(py, same && !moved)?;
        }
        // Python emptied or replaced a dict, or removed entries: the memo
        // is then what `objects` holds, with the originals `_sources` holds,
        // letting go of the others as the Python walk did.
        let rebuilds = !same || (!flushes && moved);
        let len = objects.len();
        let changed = rebuilds || len != self.mirrored;
        if rebuilds {
            self.rebuild(py, &objects, &sources)?;
        } else if !flushes {
            self.saw_additions(py, &sources)?;
            if len > self.mirrored && !self.foreign {
                // Python added copies; the walker had none unwritten.
                self.adopt_tail(py, &objects, &sources, len - self.mirrored)?;
            }
        }
        if overwrites && !self.foreign {
            self.recheck(py, &objects, &sources)?;
        }
        if changed {
            if self.memo.len() != len && !self.foreign {
                // A copy dropped mid-dict.
                self.rebuild(py, &objects, &sources)?;
            }
            self.mirrored = self.memo.len();
        }
        self.saw_sources(py, &sources)?;
        self.mirror = Some((objects.unbind(), sources.unbind()));
        Ok(())
    }

    /// Whether `objects` and `_sources` still hold, where the walker last
    /// saw each end, the key it saw there: whether Python code only added
    /// entries after them.
    pub(crate) fn follows(
        &self,
        py: Python<'_>,
        objects: &Bound<'_, PyDict>,
        sources: &Bound<'_, PyDict>,
    ) -> PyResult<bool> {
        let last = (self.mirrored.checked_sub(1))
            .and_then(|at| self.memo.iter_from(at).next())
            .map(|(key, _)| key);
        if key_ending(py, objects, self.mirrored)?.and_then(|key| id_of(&key)) != last {
            return Ok(false);
        }
        Ok(
            match (
                key_ending(py, sources, self.sources_mirrored)?,
                &self.sources_last,
            ) {
                // A sync that raised after its rebuild leaves no newest key
                // for a count it saw: the dict may have shrunk since.
                (None, None) => self.sources_mirrored == 0,
                (Some(key), Some(last)) => key.is(last.bind(py)),
                _ => false,
            },
        )
    }

    /// Notes how many entries `_sources` holds and the newest key, which
    /// [`State::follows`] looks for there later.
    pub(crate) fn saw_sources(
        &mut self,
        py: Python<'_>,
        sources: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let len = sources.len();
        if len != self.sources_mirrored || (len > 0 && self.sources_last.is_none()) {
            self.sources_last = newest(py, sources, 1)?.pop().map(Bound::unbind);
            self.sources_mirrored = len;
        }
        Ok(())
    }

    /// Checks the keys Python code appended to `_sources` since the walker
    /// last saw it, before the walker looks an original up there, which may
    /// run the own methods of a key not of a plain kind.
    pub(crate) fn saw_additions(
        &mut self,
        py: Python<'_>,
        sources: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        if sources.len() > self.sources_mirrored {
            let added = newest(py, sources, sources.len() - self.sources_mirrored)?;
            self.foreign |= !added.iter().all(|key| self.s.plain_key(key));
        }
        Ok(())
    }

    /// Whether the memo holds copies, or replacements, `ctx.objects` lacks,
    /// or the walker owes `ctx._sources` originals.
    pub(crate) fn unwritten(&self) -> bool {
        self.memo.len() > self.mirrored || !self.dirty.is_empty() || !self.owed.is_empty()
    }

    /// [`State::sync`] with the dicts the context holds after a hand-back,
    /// read from its instance dict; `false` if they are not exact dicts or
    /// the context is no longer one the walker mirrors, which the next
    /// call's refresh then reports.
    pub(crate) fn resync(&mut self, py: Python<'_>, ctx: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Some(attributes) = attributes(py, self.s, &self.ctx_type, ctx)? else {
            return Ok(false);
        };
        let objects = attributes.get_item(intern!(py, "objects"))?;
        let sources = attributes.get_item(intern!(py, "_sources"))?;
        match (
            objects.map(Bound::cast_into_exact::<PyDict>),
            sources.map(Bound::cast_into_exact::<PyDict>),
        ) {
            (Some(Ok(objects)), Some(Ok(sources))) => {
                self.sync(py, objects, sources, true)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Makes the memo exactly the copies `objects` holds, in its order.
    pub(crate) fn rebuild(
        &mut self,
        py: Python<'_>,
        objects: &Bound<'_, PyDict>,
        sources: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        self.memo.clear();
        self.truncated.clear();
        self.dirty.clear();
        self.sources_last = None;
        // Looking an original up in `_sources` may run a key's own methods.
        self.foreign = !self.s.plain_dict(sources);
        let keys = objects.keys().iter().collect();
        self.adopt(py, objects, sources, keys)
    }

    /// Adopts the last `count` copies Python added to `objects`, in order.
    fn adopt_tail(
        &mut self,
        py: Python<'_>,
        objects: &Bound<'_, PyDict>,
        sources: &Bound<'_, PyDict>,
        count: usize,
    ) -> PyResult<()> {
        let mut keys = newest(py, objects, count)?;
        keys.reverse();
        self.adopt(py, objects, sources, keys)
    }

    /// Moves the copies under `keys` into the memo, with their originals,
    /// if every key is an id. The Python
    /// walk looks its copies up by id: a key of any other kind, one equal
    /// to an id such as a float or one whose own methods compare it, only
    /// the Python walk looks up as it would, so the context goes to it, and
    /// checking runs no code.
    fn adopt(
        &mut self,
        py: Python<'_>,
        objects: &Bound<'_, PyDict>,
        sources: &Bound<'_, PyDict>,
        keys: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let mut ids = Vec::with_capacity(keys.len());
        for key in &keys {
            match id_of(key) {
                Some(id) => ids.push(id),
                None => self.foreign = true,
            }
        }
        if self.foreign {
            return Ok(());
        }
        for (key, id) in keys.iter().zip(ids) {
            let Some(captured) = objects.get_item(key)? else {
                continue;
            };
            let original = match sources.get_item(key)? {
                Some(original) => original,
                None => py.None().into_bound(py),
            };
            if type_address(&captured) == self.s.truncated_type && !captured.is(&original) {
                self.truncated.push(id);
            }
            let entry = Entry {
                slot: Slot::Value(captured.unbind()),
                original: original.unbind(),
            };
            self.memo.insert(id, entry);
        }
        Ok(())
    }

    /// Adopts the copies the Python walk put in place of depth placeholders:
    /// its `_mapping` captures a mapping omitted deeper down when it meets
    /// the mapping again as a root, under the same id.
    fn recheck(
        &mut self,
        py: Python<'_>,
        objects: &Bound<'_, PyDict>,
        sources: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let mut keys = std::mem::take(&mut self.truncated);
        keys.sort_unstable();
        keys.dedup();
        for key in keys {
            let Some(Entry {
                slot: Slot::Value(cached),
                original,
            }) = self.memo.get(key)
            else {
                continue;
            };
            let (cached, original) = (cached.bind(py).clone(), original.clone_ref(py));
            if type_address(&cached) != self.s.truncated_type {
                continue;
            }
            // Only a depth placeholder is ever replaced; the walker made
            // it, so its reason is an exact str in an exact dict.
            let reason = cached.getattr(intern!(py, "reason"))?;
            if type_address(&reason) != self.s.str_ || reason.ne(intern!(py, "max_depth"))? {
                continue;
            }
            let current = match objects.get_item(key)? {
                Some(current) if !current.is(&cached) => current,
                _ => {
                    self.truncated.push(key);
                    continue;
                }
            };
            let original = match sources.get_item(key)? {
                Some(source) => source.unbind(),
                None => original,
            };
            if type_address(&current) == self.s.truncated_type {
                self.truncated.push(key);
            }
            let entry = Entry {
                slot: Slot::Value(current.unbind()),
                original,
            };
            self.memo.insert(key, entry);
        }
        Ok(())
    }
}

/// The id a memo key stands for: an exact int, which hashes and compares
/// without running code, in `usize` range.
pub(crate) fn id_of(key: &Bound<'_, PyAny>) -> Option<usize> {
    match key.is_exact_instance_of::<PyInt>() {
        true => key.extract::<usize>().ok(),
        false => None,
    }
}

/// The key `dict` holds at index `count - 1`, behind its newer ones; none
/// if `count` is zero or beyond its length.
fn key_ending<'py>(
    py: Python<'py>,
    dict: &Bound<'py, PyDict>,
    count: usize,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match count {
        0 => Ok(None),
        _ if dict.len() < count => Ok(None),
        _ => Ok(newest(py, dict, dict.len() + 1 - count)?.pop()),
    }
}

/// The last `count` keys of `dict`, newest first.
fn newest<'py>(
    py: Python<'py>,
    dict: &Bound<'py, PyDict>,
    count: usize,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let newest_first = dict.call_method0(intern!(py, "__reversed__"))?;
    let mut keys = Vec::with_capacity(count);
    for key in newest_first.try_iter()?.take(count) {
        keys.push(key?);
    }
    Ok(keys)
}
