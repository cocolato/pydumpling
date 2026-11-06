//! Keeps the walker's memo and the context's `objects` in step.
//!
//! The Python walk memoizes every copy in `ctx.objects`, keyed by the
//! original's id, and keeps the original alive in `ctx._sources`. The
//! walker memoizes in [`crate::memo::Memo`] and writes those dicts lazily:
//! the oldest [`State::mirrored`] entries are in `ctx.objects`, in the same
//! order, and the rest are written only before Python code reads them, on a
//! hand-back or a copy of the context. Python adds copies to `ctx.objects`
//! by appending, so afterwards the walker adopts just the new tail; one
//! that replaces or empties the dict resets the memo, as it resets the
//! Python walk's.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::statics::type_address;
use crate::walker::{Entry, Halt, Slot, State, Walker};

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
        let len = objects.len();
        let same = self
            .mirror
            .as_ref()
            .is_some_and(|(known, _)| known.bind(py).is(&objects));
        let mut unwritten = self.memo.len() > self.mirrored || !self.dirty.is_empty();
        if !same && unwritten {
            // The dicts the walker mirrored, replaced or another context's,
            // get every copy first, as the Python walk wrote each at once.
            self.flush(py)?;
            unwritten = false;
        }
        if same && len == self.mirrored {
            // Python added nothing; the walker's newer copies stay unwritten.
            if overwrites {
                self.recheck(py, &objects, &sources)?;
            }
        } else {
            if same && len > self.mirrored && !unwritten {
                self.adopt_tail(py, &objects, &sources, len - self.mirrored)?;
            } else {
                if same && len > self.mirrored {
                    // Python added copies while the walker held newer ones:
                    // keep both, then take the dict's order.
                    self.write(py, &objects, &sources, true)?;
                }
                // Otherwise Python emptied or replaced the memo, which then
                // is whatever `objects` holds.
                self.rebuild(py, &objects, &sources)?;
            }
            if overwrites {
                self.recheck(py, &objects, &sources)?;
            }
            if self.memo.len() != objects.len() {
                // Keys that are not ids, or a copy dropped mid-dict.
                self.rebuild(py, &objects, &sources)?;
            }
            self.mirrored = self.memo.len();
        }
        self.mirror = Some((objects.unbind(), sources.unbind()));
        Ok(())
    }

    /// [`State::sync`] with the context's current dicts; `false` if they
    /// are not exact dicts, which the next call's refresh then reports.
    pub(crate) fn resync(&mut self, py: Python<'_>, ctx: &Bound<'_, PyAny>) -> PyResult<bool> {
        let objects = ctx.getattr(intern!(py, "objects"))?;
        let sources = ctx.getattr(intern!(py, "_sources"))?;
        match (
            objects.cast_into_exact::<PyDict>(),
            sources.cast_into_exact::<PyDict>(),
        ) {
            (Ok(objects), Ok(sources)) => {
                self.sync(py, objects, sources, true)?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Writes the memo entries `ctx.objects` lacks into it and its sources
    /// into `ctx._sources`, so the Python walk sees every copy.
    pub(crate) fn flush(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some((objects, sources)) = self
            .mirror
            .as_ref()
            .map(|(objects, sources)| (objects.bind(py).clone(), sources.bind(py).clone()))
        else {
            return Ok(());
        };
        self.write(py, &objects, &sources, false)?;
        if objects.len() != self.memo.len() {
            // A placeholder only the Python walk can name stayed unwritten,
            // or Python held a copy of the same original.
            self.rebuild(py, &objects, &sources)?;
        }
        self.mirrored = self.memo.len();
        Ok(())
    }

    /// Writes the mirrored entries the walk replaced and the unmirrored
    /// ones, building pending placeholders first; `missing` keeps any copy
    /// Python already holds for an unmirrored entry's original.
    fn write(
        &mut self,
        py: Python<'_>,
        objects: &Bound<'_, PyDict>,
        sources: &Bound<'_, PyDict>,
        missing: bool,
    ) -> PyResult<()> {
        let pending: Vec<usize> = self
            .memo
            .iter_from(self.mirrored)
            .filter(|(_, entry)| matches!(entry.slot, Slot::Recursive))
            .map(|(key, _)| key)
            .collect();
        for key in pending {
            if let Err(Halt::Py(err)) = self.placeholder(py, key) {
                return Err(err);
            }
        }
        let set = |key: usize, entry: &Entry, keep: bool| -> PyResult<()> {
            let Slot::Value(captured) = &entry.slot else {
                return Ok(());
            };
            let id = key.into_pyobject(py)?;
            if keep && objects.contains(&id)? {
                return Ok(());
            }
            objects.set_item(&id, captured)?;
            sources.set_item(&id, &entry.original)
        };
        for key in std::mem::take(&mut self.dirty) {
            if let Some(entry) = self.memo.get(key) {
                set(key, entry, false)?;
            }
        }
        for (key, entry) in self.memo.iter_from(self.mirrored) {
            set(key, entry, missing)?;
        }
        Ok(())
    }

    /// Makes the memo exactly the copies `objects` holds, in its order.
    fn rebuild(
        &mut self,
        py: Python<'_>,
        objects: &Bound<'_, PyDict>,
        sources: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        self.memo.clear();
        self.truncated.clear();
        self.dirty.clear();
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
        let newest_first = objects.call_method0(intern!(py, "__reversed__"))?;
        let mut keys = Vec::with_capacity(count);
        for key in newest_first.try_iter()?.take(count) {
            keys.push(key?);
        }
        keys.reverse();
        self.adopt(py, objects, sources, keys)
    }

    /// Moves the copies under `keys` into the memo, with their originals.
    fn adopt(
        &mut self,
        py: Python<'_>,
        objects: &Bound<'_, PyDict>,
        sources: &Bound<'_, PyDict>,
        keys: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        for key in keys {
            // Only an id is ever looked up; anything else stays in the dict.
            let (Ok(id), Some(captured)) = (key.extract::<usize>(), objects.get_item(&key)?) else {
                continue;
            };
            let original = sources
                .get_item(&key)?
                .unwrap_or_else(|| py.None().into_bound(py));
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

impl Drop for Walker {
    /// A walker dropped while its memo is still shared, as by a context
    /// that lives on or a copy of it, writes the copies `objects` lacks.
    fn drop(&mut self) {
        let state = &mut self.state;
        if state.memo.len() <= state.mirrored {
            return;
        }
        Python::try_attach(|py| {
            let Some((objects, _)) = &state.mirror else {
                return;
            };
            // Whether anything but the walker holds `objects`: compared
            // with a dict only the walker holds, so the count is right
            // however the call passes its argument.
            let shared = py
                .import("sys")
                .and_then(|sys| sys.getattr("getrefcount"))
                .and_then(|count| {
                    let alone = PyDict::new(py);
                    let held: isize = count.call1((objects.bind(py),))?.extract()?;
                    let baseline: isize = count.call1((&alone,))?.extract()?;
                    Ok(held > baseline)
                });
            // A destructor cannot raise; the dicts keep what was written.
            if shared.unwrap_or(false) && state.flush(py).is_err() {
                state.memo.clear();
            }
        });
    }
}
