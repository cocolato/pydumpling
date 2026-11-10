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

use crate::statics::type_address;
use crate::types::{placeholder_like, plain_reason, reads_plainly};
use crate::walker::{Entry, Slot, State};

impl State {
    /// Brings the memo in line with `objects`, which Python code may have
    /// added to, emptied, or replaced since the last sync, and adopts the
    /// depth placeholders a Python walk replaced in place: one a hand-back
    /// ran since the last recheck, or, once a copy shares the dicts, one
    /// it may run at any time.
    pub(crate) fn sync(
        &mut self,
        py: Python<'_>,
        objects: Bound<'_, PyDict>,
        sources: Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let rechecks = self.eager || self.unchecked;
        let mut passes = 0_u64;
        loop {
            self.take_in(py, &objects, &sources)?;
            // Last: the recheck may raise (out of memory, at CPython's C
            // recursion limit, or from the `__eq__` of a key an unnoticed
            // removal let in that hashes like an id), when the walker has
            // then taken in all else it found, and stays unchecked.
            if !rechecks || self.foreign {
                return Ok(());
            }
            // The Python walk runs the finalizers of a placeholder it
            // replaces nested: the recursion limit ends a chain of them,
            // except on 3.12, where CPython's trashcan frees the
            // placeholders one after another. Past it, the next sync
            // carries on.
            if passes > 0 && passes >= self.s.recursion_limit.bind(py).call0()?.extract()? {
                return Ok(());
            }
            self.recheck(py, &objects, &sources)?;
            passes += 1;
            // Unless a placeholder it let go of had finalizers that wrote
            // there, as they ran when the Python walk replaced it.
            if objects.len() == self.mirrored && sources.len() == self.sources_mirrored {
                self.unchecked = false;
                return Ok(());
            }
        }
    }

    /// The sync up to its recheck.
    fn take_in(
        &mut self,
        py: Python<'_>,
        objects: &Bound<'_, PyDict>,
        sources: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let same = match &self.mirror {
            Some((known, kept)) => known.bind(py).is(objects) && kept.bind(py).is(sources),
            None => false,
        };
        // Python code removed entries: a dict with fewer than the walker
        // saw has none where it saw the last. Removals alongside additions,
        // as another Python walk makes during a hand-back, show in no count,
        // but move that entry; only a removal that leaves it where it was
        // goes unnoticed.
        let moved = same && !self.follows(py, objects, sources)?;
        let flushes = self.unwritten() && (!same || moved || objects.len() != self.mirrored);
        if flushes {
            // The walker's copies go into the dicts it mirrored, after what
            // Python added; the flush then takes the memo from `objects`.
            self.flush(py, same && !moved)?;
        }
        if !same {
            // Python let go of a dict it replaced then, and its finalizers,
            // which may write into `objects`, ran: they run before it is read.
            self.mirror = None;
        }
        // Python emptied or replaced a dict, or removed entries: the memo
        // is then what `objects` holds, with the originals `_sources` holds,
        // letting go of the others as the Python walk did.
        let rebuilds = !same || (!flushes && moved);
        let len = objects.len();
        let changed = rebuilds || len != self.mirrored;
        if rebuilds {
            self.rebuild(py, objects, sources)?;
        } else if !flushes {
            self.saw_additions(py, sources)?;
            if len > self.mirrored && !self.foreign {
                // Python added copies; the walker had none unwritten.
                self.adopt_tail(py, objects, sources, len - self.mirrored)?;
            }
        }
        if changed {
            if self.memo.len() != len && !self.foreign {
                // A copy dropped mid-dict.
                self.rebuild(py, objects, sources)?;
            }
            self.mirrored = self.memo.len();
        }
        self.saw_sources(py, sources)?;
        self.mirror = Some((objects.clone().unbind(), sources.clone().unbind()));
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
                // A sync or write that failed after its rebuild, before the
                // walker noted `_sources`, leaves no newest key for a count
                // it saw: the dict may have shrunk since.
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
    /// if every key is an id. The Python walk looks its copies up by id: a
    /// key of any other kind, one equal to an id such as a float or one
    /// whose own methods compare it, only the Python walk looks up as it
    /// would, so the context goes to it, and checking runs no code.
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
            if placeholder_like(py, self.s, &captured)? {
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
        if self.truncated.is_empty() {
            return Ok(());
        }
        let mut keys = std::mem::take(&mut self.truncated);
        keys.sort_unstable();
        keys.dedup();
        // Only an exact `TruncatedValue`'s reason is read.
        let class = self.s.truncated.bind(py);
        let plain = match reads_plainly(py, self.s, class, intern!(py, "reason")) {
            Ok(plain) => plain,
            Err(err) => {
                self.truncated = keys;
                return Err(err);
            }
        };
        for (at, &key) in keys.iter().enumerate() {
            if let Err(err) = self.take_up(py, objects, sources, key, plain) {
                // It and the ones not yet checked stay noted for the next.
                self.truncated.extend_from_slice(&keys[at..]);
                return Err(err);
            }
        }
        Ok(())
    }

    /// Adopts what `objects` holds for `key` if the Python walk put it in
    /// place of the memo's depth placeholder, and keeps a placeholder that
    /// may still be replaced noted. `plain` if `TruncatedValue` reads
    /// `reason` only from the instance dict.
    fn take_up(
        &mut self,
        py: Python<'_>,
        objects: &Bound<'_, PyDict>,
        sources: &Bound<'_, PyDict>,
        key: usize,
        plain: bool,
    ) -> PyResult<()> {
        let Some(Entry {
            slot: Slot::Value(cached),
            original,
        }) = self.memo.get(key)
        else {
            return Ok(());
        };
        let (cached, original) = (cached.bind(py).clone(), original.clone_ref(py));
        // The Python walk's `_mapping` replaces only a `TruncatedValue`, of
        // a subclass too: a key the walker's own copy took since, or whose
        // placeholder application code gave another class, is let go of.
        if !placeholder_like(py, self.s, &cached)? {
            return Ok(());
        }
        let current = match objects.get_item(key)? {
            Some(current) if !current.is(&cached) => current,
            _ => {
                // Still in place. Only a depth placeholder is ever replaced,
                // so one whose reason reads as another is let go of.
                // Application code may hold it, and may have changed it or
                // its class since: one of a subclass, or whose reason the
                // walker cannot read without running code or finds missing,
                // stays noted, as the Python walk may yet replace it.
                if plain && type_address(&cached) == self.s.truncated_type {
                    if let Some(reason) = plain_reason(py, self.s, &cached)? {
                        if reason.ne(intern!(py, "max_depth"))? {
                            return Ok(());
                        }
                    }
                }
                self.truncated.push(key);
                return Ok(());
            }
        };
        let original = match sources.get_item(key)? {
            Some(source) => source.unbind(),
            None => original,
        };
        if placeholder_like(py, self.s, &current)? {
            self.truncated.push(key);
        }
        let entry = Entry {
            slot: Slot::Value(current.unbind()),
            original,
        };
        self.memo.insert(key, entry);
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
