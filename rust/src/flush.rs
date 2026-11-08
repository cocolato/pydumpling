//! Writes the copies the walker memoized into `ctx.objects` and
//! `ctx._sources`, where the Python walk would have put them.
//!
//! The walker writes lazily (see [`crate::mirror`]): until Python code may
//! read the dicts, on a hand-back, a copy of the context or its release
//! ([`crate::release`]), the newest memo entries are only in the memo, and
//! the writes into `_sources` a release left out only in the walker's
//! [`State::owed`], which go in first. Writing them then must give both
//! dicts the order the Python walk gave them, though Python code may have
//! added entries of its own in the meantime: those came later, so they are
//! taken out and put back after the walker's, unless a key whose comparison
//! is its own may meet theirs, when they stay ahead (but for a key a removal
//! the walker does not notice lets in: see `rust/migration/BLUEPRINT.md`).
//! An entry the Python walk wrote in place of one the dict held, Python
//! code may have removed since, and it stays removed.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::memo::AddressMap;
use crate::mirror::id_of;
use crate::statics::Statics;
use crate::walker::{Entry, Halt, Slot, State};

/// Entries taken out of a dict, in order, to go back at its end.
type Later<'py> = Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>;

/// Which entry of a dict a memo entry is written over.
#[derive(Clone, Copy)]
pub(crate) enum Put {
    /// Any: the Python walk wrote it after the dict's own.
    Always,
    /// None: Python code wrote its own entry for it later, which stays.
    IfAbsent,
    /// Only the one it replaces in place, which Python code may have
    /// removed since.
    IfPresent,
}

impl Put {
    fn write(
        self,
        dict: &Bound<'_, PyDict>,
        id: &Bound<'_, PyAny>,
        value: &Py<PyAny>,
    ) -> PyResult<()> {
        let write = match self {
            Put::Always => true,
            Put::IfAbsent => !dict.contains(id)?,
            Put::IfPresent => dict.contains(id)?,
        };
        if write {
            dict.set_item(id, value)?;
        }
        Ok(())
    }
}

/// A write into `_sources` that a release, writing only `ctx.objects`,
/// left out, and that the walker's next write makes, as it would have
/// made it then: see [`crate::release`].
pub(crate) struct Owed {
    /// The id it goes in under.
    pub(crate) id: usize,
    /// The original, as the walk held it.
    pub(crate) original: Py<PyAny>,
    /// Whether it goes over the entry `_sources` holds, only if one is
    /// there; otherwise it goes in as a new entry does: in front of what
    /// Python code added since, unless those stay in place, when after
    /// them, and never over Python's own entry at its id.
    pub(crate) in_place: bool,
}

impl Owed {
    fn clone_ref(&self, py: Python<'_>) -> Self {
        let original = self.original.clone_ref(py);
        Self { original, ..*self }
    }
}

impl State {
    /// Writes the memo entries the mirrored `objects` lacks into it and
    /// their originals into the mirrored `_sources`: the dicts the copies
    /// were made for, as every sync with other dicts flushes first.
    /// `checked` if the caller has just checked where Python code left
    /// each dict's last mirrored entry, as a sync does before it flushes.
    pub(crate) fn flush(&mut self, py: Python<'_>, checked: bool) -> PyResult<()> {
        let Some((objects, sources)) = self
            .mirror
            .as_ref()
            .map(|(objects, sources)| (objects.bind(py).clone(), sources.bind(py).clone()))
        else {
            return Ok(());
        };
        // Entries Python removed while it added others show in no count.
        let moved = !checked && !self.follows(py, &objects, &sources)?;
        let replaced = self.write(py, &objects, &sources, moved)?;
        if replaced || moved || objects.len() != self.memo.len() {
            // Python added copies of its own, replaced the walker's, added
            // a key that is not an id, or removed entries.
            self.rebuild(py, &objects, &sources)?;
        }
        self.mirrored = self.memo.len();
        self.saw_sources(py, &sources)
    }

    /// Writes the mirrored entries the walk replaced and the unmirrored ones,
    /// building pending placeholders first. Entries Python added to either
    /// dict since the walker last wrote it are taken out and put back after
    /// the walker's, replacing in place a copy of the same original, as when
    /// the Python walk wrote its copies at once; `true` if one replaced a copy
    /// of the walker's in `objects`, if they had to stay in place, or if
    /// Python removed originals from `_sources`, which the memo then lets go
    /// of, as `moved` says it may have. An entry the walk would have replaced
    /// in place goes in only if the dict still holds one.
    fn write(
        &mut self,
        py: Python<'_>,
        objects: &Bound<'_, PyDict>,
        sources: &Bound<'_, PyDict>,
        moved: bool,
    ) -> PyResult<bool> {
        let pending: Vec<usize> = self
            .memo
            .iter_from(self.mirrored)
            .filter(|(_, entry)| matches!(entry.slot, Slot::Recursive))
            .map(|(key, _)| key)
            .collect();
        for key in pending {
            // Naming an exact tuple or frozenset runs no Python code, so only
            // a recursion overflow leaves one unbuilt; it is then left out,
            // and the memo is rebuilt from `objects` without it.
            if let Err(Halt::Py(err)) = self.placeholder(py, key) {
                return Err(err);
            }
        }
        let removed = moved || sources.len() < self.sources_mirrored;
        let (kept_objects, kept_sources) = match removed {
            // Python code removed entries since the walker last wrote.
            true => {
                self.prune_sources(sources);
                self.kept_entries(py, objects, sources)
            }
            false => (self.mirrored, self.sources_mirrored),
        };
        // The walker checked each key it read of the dicts, and a removal it
        // noticed may have let Python's later keys into the part it saw.
        let whole = removed || self.foreign;
        let later_objects = take_tail(py, self.s, objects, kept_objects, whole)?;
        let later_sources = take_tail(py, self.s, sources, kept_sources, whole);
        // Where Python's entries stay in place, one it wrote for an original
        // the walker copied came later than the walker's copy, and stays.
        let new = |all: bool| if all { Put::Always } else { Put::IfAbsent };
        let written = match &later_sources {
            Ok(later_sources) => {
                let new = (new(later_objects.is_some()), new(later_sources.is_some()));
                self.write_entries(py, objects, sources, new)
            }
            Err(_) => Ok(()),
        };
        // Python's entries go back after the walker's, though a write
        // failed: Python code wrote them, and the walker must not lose them.
        let replaced = match later_objects {
            Some(later) => put_back(objects, later),
            None => Ok(true),
        };
        let put_sources = match later_sources {
            Ok(Some(later)) => {
                let keys: Vec<_> = later.iter().map(|(key, _)| key.clone()).collect();
                let put = put_back(sources, later);
                self.note_sources(keys);
                put.map(drop)
            }
            Ok(None) => {
                // Python's entries stayed where they are: each is noted, and
                // what the walker noted before stays, as an original a write
                // that failed has yet to make in place.
                let keys: Vec<_> = sources.keys().iter().collect();
                self.note_sources(keys);
                Ok(())
            }
            Err(err) => Err(err),
        };
        written?;
        put_sources?;
        Ok(replaced? || removed)
    }

    /// Writes what the walker owes `_sources`, then the memo entries
    /// `objects` lacks and the ones the walk replaced in place, each
    /// original before its copy, so that a write that fails between them
    /// leaves no copy whose original is held nowhere. `new` is how each
    /// dict takes an entry the Python walk added. While a release owes
    /// `_sources` what goes there, [`State::owing`] notes each such write.
    fn write_entries(
        &mut self,
        py: Python<'_>,
        objects: &Bound<'_, PyDict>,
        sources: &Bound<'_, PyDict>,
        new: (Put, Put),
    ) -> PyResult<()> {
        // What a release left out came before every entry here.
        for owed in &self.owed {
            let id = owed.id.into_pyobject(py)?.into_any();
            let into = if owed.in_place { Put::IfPresent } else { new.1 };
            into.write(sources, &id, &owed.original)?;
            if let Some(owing) = &mut self.owing {
                owing.push(owed.clone_ref(py));
            }
        }
        let owing = &mut self.owing;
        // `in_place` for an original that goes over the entry `_sources`
        // holds, only if one is there.
        let mut put = |key: usize, entry: &Entry, into: Put, in_place: bool| -> PyResult<()> {
            let Slot::Value(captured) = &entry.slot else {
                return Ok(());
            };
            let id = key.into_pyobject(py)?;
            let source = if in_place { Put::IfPresent } else { new.1 };
            source.write(sources, &id, &entry.original)?;
            if let Some(owing) = owing.as_mut() {
                let original = entry.original.clone_ref(py);
                owing.push(Owed {
                    id: key,
                    original,
                    in_place,
                });
            }
            into.write(objects, &id, captured)
        };
        let mut dirty = self.dirty.iter().copied().peekable();
        let held = &self.sources_only;
        for (at, (key, entry)) in (self.mirrored..).zip(self.memo.iter_from(self.mirrored)) {
            // A copy the walk replaced in place keeps its place in
            // `objects`, and goes into a `_sources` that lacked it as a new
            // entry, after the copies made before the replacement, as the
            // Python walk appended it there.
            while let Some((replaced, _, was)) = dirty.next_if(|&(_, when, _)| when <= at) {
                if let Some(entry) = self.memo.get(replaced) {
                    put(replaced, entry, Put::IfPresent, was)?;
                }
            }
            put(key, entry, new.0, held.contains_key(&key))?;
        }
        for (replaced, _, was) in dirty {
            if let Some(entry) = self.memo.get(replaced) {
                put(replaced, entry, Put::IfPresent, was)?;
            }
        }
        // Only now: a write that fails leaves all of it to the next, as it
        // does the memo entries past `mirrored`, and writing any again puts
        // it where it was.
        self.owed.clear();
        self.dirty.clear();
        Ok(())
    }

    /// How many entries of `objects` and `_sources` come before the ones
    /// Python code added since the walker last wrote them, when it removed
    /// some too, so that the counts no longer say.
    fn kept_entries(
        &self,
        py: Python<'_>,
        objects: &Bound<'_, PyDict>,
        sources: &Bound<'_, PyDict>,
    ) -> (usize, usize) {
        let unwritten: AddressMap<()> = (self.memo.iter_from(self.mirrored))
            .map(|(key, _)| (key, ()))
            .collect();
        // A release whose write failed let go of the memo (see
        // `crate::release`): every copy `objects` holds then the walker took
        // for its own, its original with it. The keys are read in order,
        // which, unlike looking an id up, runs no key's own methods.
        let forgotten: AddressMap<()> = match self.memo.is_empty() {
            true => (objects.keys().iter())
                .filter_map(|key| id_of(&key))
                .map(|id| (id, ()))
                .collect(),
            false => AddressMap::default(),
        };
        // Python wrote any entry it holds for a copy the walker has yet to
        // write after the last sync, which flushed what it found.
        let written = |id: usize| {
            (self.memo.contains(id) && !unwritten.contains_key(&id)) || forgotten.contains_key(&id)
        };
        let last_copy = (self.mirrored.checked_sub(1))
            .and_then(|at| self.memo.iter_from(at).next())
            .map(|(key, _)| key);
        let last_source = self.sources_last.as_ref().map(|last| last.bind(py));
        let kept_objects = kept_before(
            objects,
            self.mirrored,
            |key| last_copy.is_some() && id_of(key) == last_copy,
            // A key that is not an id sends the context to the Python walk.
            |key| id_of(key).is_some_and(written),
        );
        // Python wrote any entry it holds where the walker owes a new one,
        // or appends the original of a copy the walk replaced in place.
        let appended = self.dirty.iter().filter(|&&(_, _, was)| !was);
        let owed: AddressMap<()> = (self.owed.iter())
            .filter(|owed| !owed.in_place)
            .map(|owed| owed.id)
            .chain(appended.map(|&(id, _, _)| id))
            .map(|id| (id, ()))
            .collect();
        let original = |id: usize| written(id) && !owed.contains_key(&id);
        let kept_sources = kept_before(
            sources,
            self.sources_mirrored,
            |key| last_source.is_some_and(|last| key.is(last)),
            // One that is not an id the walker leaves where it is.
            |key| id_of(key).is_none_or(|id| original(id) || self.sources_only.contains_key(&id)),
        );
        (kept_objects, kept_sources)
    }
}

/// How many entries `dict` holds before the ones Python code added since the
/// walker last wrote its first `part`: no more than those, as Python removes
/// entries and appends its own, none after the entry the walker wrote last,
/// which `last` finds, and none from the first whose key `known` does not
/// know, as Python may have removed that entry, or put it back after others.
fn kept_before(
    dict: &Bound<'_, PyDict>,
    part: usize,
    last: impl Fn(&Bound<'_, PyAny>) -> bool,
    known: impl Fn(&Bound<'_, PyAny>) -> bool,
) -> usize {
    let mut kept = 0;
    for (key, _) in dict.iter().take(part) {
        if !known(&key) {
            break;
        }
        kept += 1;
        if last(&key) {
            break;
        }
    }
    kept
}

/// Puts the entries [`take_tail`] took out back at the end of `dict`, each
/// one though putting another fails; whether `dict` held a key of theirs
/// again, as the walker wrote it there.
fn put_back(dict: &Bound<'_, PyDict>, later: Later<'_>) -> PyResult<bool> {
    let mut again = false;
    let mut result = Ok(());
    for (key, value) in later {
        let put = dict.contains(&key).and_then(|held| {
            again |= held;
            dict.set_item(&key, value)
        });
        result = result.and(put);
    }
    result.map(|()| again)
}

/// Removes and returns, in order, the entries `dict` holds beyond its first
/// `keep`, when every key they may meet has a builtin hash and comparison:
/// looking a key up compares it with each key of the same hash, so only
/// then do taking the entries out and putting them back run no application
/// code, which could fail between the two. Those are their own keys, and,
/// if `whole`, every key `dict` holds; otherwise the walker checked the
/// others as it read them, but for one a removal it did not notice let in.
/// If one is not, `None`, and they stay in place.
fn take_tail<'py>(
    py: Python<'py>,
    s: &Statics,
    dict: &Bound<'py, PyDict>,
    keep: usize,
    whole: bool,
) -> PyResult<Option<Later<'py>>> {
    let len = dict.len();
    if len <= keep {
        return Ok(Some(Vec::new()));
    }
    if whole && !s.plain_dict(dict) {
        return Ok(None);
    }
    let newest_first = dict.call_method0(intern!(py, "__reversed__"))?;
    let mut later = Vec::with_capacity(len - keep);
    for key in newest_first.try_iter()?.take(len - keep) {
        let key = key?;
        if !s.plain_key(&key) {
            return Ok(None);
        }
        later.push(key);
    }
    let mut entries = Vec::with_capacity(later.len());
    for key in later.into_iter().rev() {
        let Some(value) = dict.get_item(&key)? else {
            continue;
        };
        dict.del_item(&key)?;
        entries.push((key, value));
    }
    Ok(Some(entries))
}
