//! Writes the copies the walker memoized into `ctx.objects` and
//! `ctx._sources`, where the Python walk would have put them.
//!
//! The walker writes lazily (see [`crate::mirror`]): until Python code may
//! read the dicts, on a hand-back, a copy of the context or its release
//! ([`crate::release`]), the newest memo entries are only in the memo, and
//! the writes into `_sources` a release left out only in the walker's
//! [`State::owed`], which go in first. A write appends them in the memo's
//! order, each original before its copy, and writes the copies the walk
//! replaced in place over the ones `objects` holds. What Python code added
//! to either dict in the meantime stays ahead of them; when it changed
//! `objects`, the walker then takes its memo from that dict (see
//! "Accepted, documented differences" in `rust/DESIGN.md`).

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::walker::{Entry, Halt, Slot, State};

/// A write into `_sources` that a release, writing only `ctx.objects`,
/// left out, and that the walker's next write makes, as it would have
/// made it then: see [`crate::release`].
pub(crate) struct Owed {
    /// The id it goes in under.
    pub(crate) id: usize,
    /// The original, as the walk held it.
    pub(crate) original: Py<PyAny>,
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
        // Python removed entries, maybe while it added others, which shows
        // in no count.
        let moved = !checked && !self.follows(py, &objects, &sources)?;
        if !moved {
            // Otherwise the rebuild below reads every key.
            self.saw_additions(py, &sources)?;
        }
        // Python added, removed or replaced copies, or removed originals,
        // which the memo then lets go of, as the Python walk did.
        let rewritten = moved || objects.len() != self.mirrored;
        self.write(py, &objects, &sources, rewritten)?;
        if rewritten || objects.len() != self.memo.len() {
            self.rebuild(py, &objects, &sources)?;
        }
        self.mirrored = self.memo.len();
        self.saw_sources(py, &sources)
    }

    /// Builds pending placeholders, then writes what the walker owes
    /// `_sources`, the copies the walk replaced in place, and the memo
    /// entries `objects` lacks, each original before its copy, so that a
    /// write that fails between them leaves no copy whose original is held
    /// nowhere. While a release owes `_sources` what goes there,
    /// [`State::owing`] notes each such write. Where Python code rewrote
    /// `objects` (`rewritten`), a new copy goes in only if it holds none
    /// for that id:
    /// Python wrote its own later, which the Python walk then kept.
    fn write(
        &mut self,
        py: Python<'_>,
        objects: &Bound<'_, PyDict>,
        sources: &Bound<'_, PyDict>,
        rewritten: bool,
    ) -> PyResult<()> {
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
        // What a release left out came before every entry here.
        for owed in &self.owed {
            let id = owed.id.into_pyobject(py)?;
            sources.set_item(id, &owed.original)?;
            if let Some(owing) = &mut self.owing {
                owing.push(owed.clone_ref(py));
            }
        }
        let owing = &mut self.owing;
        let mut put = |key: usize, entry: &Entry, new: bool| -> PyResult<()> {
            let Slot::Value(captured) = &entry.slot else {
                return Ok(());
            };
            let id = key.into_pyobject(py)?;
            sources.set_item(&id, &entry.original)?;
            if let Some(owing) = owing.as_mut() {
                let original = entry.original.clone_ref(py);
                owing.push(Owed { id: key, original });
            }
            if !(new && rewritten && objects.contains(&id)?) {
                objects.set_item(&id, captured)?;
            }
            Ok(())
        };
        // Each goes over the entry it replaces, where it was.
        for &replaced in &self.dirty {
            if let Some(entry) = self.memo.get(replaced) {
                put(replaced, entry, false)?;
            }
        }
        for (key, entry) in self.memo.iter_from(self.mirrored) {
            put(key, entry, true)?;
        }
        // Only now: a write that fails leaves all of it to the next, as it
        // does the memo entries past `mirrored`, and writing any again puts
        // it where it was.
        self.owed.clear();
        self.dirty.clear();
        Ok(())
    }
}
