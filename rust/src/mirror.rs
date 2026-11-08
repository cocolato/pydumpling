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
//! resets the Python walk's. Entries Python code adds to either dict while
//! the walker holds unwritten copies came later than those copies, so the
//! walker first writes its copies in front of them. The ids `_sources`
//! holds without a copy are noted, since the Python walk writes an original
//! the walker copies later in their place. Entries Python removes, the
//! memo lets go of, as the Python walk then holds them nowhere; at every
//! sync the walker checks that the last entries it saw are still where it
//! saw them, as removing some while adding others shows in no count. It
//! looks originals up in `_sources` only while every key it read there is
//! of a plain kind, whose lookup runs no code; a key a removal it does not
//! notice lets in, it never reads.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt};

use crate::memo::AddressMap;
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
        let (same, same_sources) = match &self.mirror {
            Some((known, kept)) => (known.bind(py).is(&objects), kept.bind(py).is(&sources)),
            None => (false, false),
        };
        let same = same && same_sources;
        let resized = objects.len() != self.mirrored || sources.len() != self.sources_mirrored;
        // Entries Python code removed while it added others, as another
        // Python walk does during a hand-back, show in no count, but move
        // the last entry the walker saw; only a removal that leaves that
        // entry where it was goes unnoticed.
        let moved = same && !self.follows(py, &objects, &sources)?;
        let flushes = self.unwritten() && (!same || resized || moved);
        if flushes {
            // Python code replaced a dict, or added or removed entries,
            // while the walker held unwritten copies, which the Python walk
            // had written at once: they go into the dicts the walker
            // mirrored first, in front of entries Python added. A flush
            // lets go of the originals Python removed, and notes the ids
            // it added, itself.
            self.flush(py, same && !moved)?;
        }
        let (len, seen) = (objects.len(), self.sources_mirrored);
        // Python emptied or replaced a dict, or removed entries: the memo
        // is then what `objects` holds, with the originals `_sources` holds,
        // letting go of the others as the Python walk did.
        let rebuilds =
            !same || len < self.mirrored || (!flushes && (sources.len() < seen || moved));
        let added = match rebuilds || flushes || sources.len() <= seen {
            true => Vec::new(),
            false => newest(py, &sources, sources.len() - seen)?,
        };
        // Looking an original up in `_sources` may run a key's own methods.
        self.foreign |= !added.iter().all(|key| self.s.plain_key(key));
        let changed = rebuilds || len != self.mirrored;
        if rebuilds {
            self.rebuild(py, &objects, &sources)?;
        } else if len > self.mirrored && !self.foreign {
            // Python added copies; the walker had none unwritten.
            self.adopt_tail(py, &objects, &sources, len - self.mirrored)?;
        }
        if overwrites && !self.foreign {
            self.recheck(py, &objects, &sources)?;
        }
        // A flush or rebuild notes the ids `_sources` holds alone itself.
        let mut noted = rebuilds || flushes;
        if changed {
            if self.memo.len() != len && !self.foreign {
                // A copy dropped mid-dict.
                self.rebuild(py, &objects, &sources)?;
                noted = true;
            }
            self.mirrored = self.memo.len();
        }
        if !noted {
            self.note_sources(added);
        }
        if !self.unwritten() {
            self.saw_sources(py, &sources)?;
        }
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
                (None, None) => true,
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
        self.sources_only.clear();
        self.sources_last = None;
        // Looking an original up in `_sources` may run a key's own methods.
        self.foreign = !self.s.plain_dict(sources);
        let keys = objects.keys().iter().collect();
        let found = self.adopt(py, objects, sources, keys)?;
        if sources.len() > found && !self.foreign {
            // Ids only `_sources` holds; otherwise it holds the copies' alone.
            self.rescan_sources(sources);
        }
        Ok(())
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
        self.adopt(py, objects, sources, keys)?;
        Ok(())
    }

    /// Notes the ids among `keys`, `_sources` entries Python code added,
    /// that have no memo entry. A key that is not an id the Python walk
    /// never writes over; one of a kind whose own methods looking an id up
    /// may run sends the context to the Python walk.
    pub(crate) fn note_sources<'py>(&mut self, keys: impl IntoIterator<Item = Bound<'py, PyAny>>) {
        for key in keys {
            self.foreign |= !self.s.plain_key(&key);
            if let Some(id) = id_of(&key) {
                if !self.memo.contains(id) {
                    self.sources_only.insert(id, ());
                }
            }
        }
    }

    /// Notes the ids `_sources` holds without a memo entry anew.
    pub(crate) fn rescan_sources(&mut self, sources: &Bound<'_, PyDict>) {
        self.sources_only.clear();
        let keys: Vec<_> = sources.keys().iter().collect();
        self.note_sources(keys);
    }

    /// Forgets the noted ids Python code removed from `_sources` before the
    /// walker copied them, before it writes. One it copied while `_sources`
    /// held it stays noted: the Python walk wrote it there in place. The
    /// keys are read in order, which, unlike looking an id up, runs no key's
    /// own methods.
    pub(crate) fn prune_sources(&mut self, sources: &Bound<'_, PyDict>) {
        let held: AddressMap<()> = sources
            .keys()
            .iter()
            .filter_map(|key| id_of(&key))
            .map(|id| (id, ()))
            .collect();
        let memo = &self.memo;
        self.sources_only
            .retain(|id, _| memo.contains(*id) || held.contains_key(id));
    }

    /// Moves the copies under `keys` into the memo, with their originals,
    /// if every key is an id; how many of them `_sources` holds. The Python
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
    ) -> PyResult<usize> {
        let mut ids = Vec::with_capacity(keys.len());
        for key in &keys {
            match id_of(key) {
                Some(id) => ids.push(id),
                None => self.foreign = true,
            }
        }
        if self.foreign {
            return Ok(0);
        }
        let mut found = 0;
        for (key, id) in keys.iter().zip(ids) {
            let Some(captured) = objects.get_item(key)? else {
                continue;
            };
            let original = match sources.get_item(key)? {
                Some(original) => {
                    found += 1;
                    original
                }
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
        Ok(found)
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
