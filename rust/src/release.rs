//! Releasing a walker: what it writes as it goes, and what it holds for
//! CPython's cyclic collector.
//!
//! The walker writes lazily, so when it goes, the copies it holds may be
//! missing from `ctx.objects` and `ctx._sources`, where the Python walk put
//! them at once. A dict that something besides the walker holds must get
//! them. The walker writes them from its finalizer, `__del__` on the class
//! `values` makes walkers of, which CPython runs as reference counting
//! frees the walker, and, for one in a reference cycle, such as a context
//! and a captured object that refers back to it, before it clears any
//! object of the cycle and before it checks what that made reachable again.
//!
//! Writing into a dict that only objects of the cycle hold is harmless
//! where the collector tracks it, as from CPython 3.14 it tracks every
//! dict: the dict, and what goes into it, are freed with the cycle. Before
//! 3.14 the collector does not track a dict that holds nothing it tracks,
//! as the walker's dicts are until it writes them, so such a dict is not
//! among the objects it frees, and what goes into it lives on, with all it
//! refers to, until a later collection. There the walker takes the
//! context's own reference to each dict for one that goes with the cycle,
//! unless the context was freed before: [`Owner`] says which. When it so
//! writes only `objects`, it keeps the writes into `_sources` it leaves
//! out, originals and all, and makes them should the context live on and
//! be used again.
//!
//! The finalizer is the only writer. Where the C stack is spent, CPython
//! refuses to call it, on every version, and the walker's write calls
//! `dict.__reversed__`, which the same check can refuse before it writes;
//! either way the copies the walker had yet to write are lost. Nothing
//! writes them as the walker is freed: the collector may have cleared them
//! by then, and an exception may be on its way up, which a call into Python
//! would replace. Nor does anything later: CPython runs a finalizer once,
//! whether or not it refused the call.

use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyWeakrefMethods, PyWeakrefReference};
use pyo3::PyTraverseError;

use crate::memo::AddressMap;
use crate::mirror::id_of;
use crate::walker::{Slot, State};

/// An object's reference count, read without calling Python, so where the
/// C stack budget is spent too. PyO3 deprecates this safe reader for an
/// unsafe one, which this crate does not use.
pub(crate) fn refcount(object: &Bound<'_, impl Sized>) -> isize {
    #[allow(deprecated)]
    object.as_any().get_refcnt()
}

/// A weak reference to the context of the walker's latest call, whose
/// callback notes that the context was freed.
pub(crate) struct Owner {
    reference: Py<PyWeakrefReference>,
    gone: Arc<AtomicBool>,
}

impl Owner {
    /// `None` for a context that takes no weak reference.
    fn new(ctx: &Bound<'_, PyAny>) -> PyResult<Option<Self>> {
        let py = ctx.py();
        let gone = Arc::new(AtomicBool::new(false));
        let noted = Arc::clone(&gone);
        let callback =
            PyCFunction::new_closure(py, Some(c"context_released"), None, move |_, _| {
                noted.store(true, Relaxed);
            })?;
        match PyWeakrefReference::new_with(ctx, callback) {
            Ok(reference) => Ok(Some(Self {
                reference: reference.unbind(),
                gone,
            })),
            Err(err) if err.is_instance_of::<PyTypeError>(py) => Ok(None),
            Err(err) => Err(err),
        }
    }

    /// Whether the context was freed while the walker lived. CPython runs
    /// the callback when it frees the context by reference counting, or in
    /// a collection that keeps the walker; not in one that frees the walker
    /// too, which first clears the weak references it frees, with no
    /// callback, as it does the walker's own reference.
    fn freed(&self) -> bool {
        self.gone.load(Relaxed)
    }
}

impl State {
    /// Notes the context of a call, before the collector's untracked
    /// dicts: a walker that Python code moved to another context, or one
    /// whose context a finalizer brought back, takes that one for its own.
    pub(crate) fn arm(&mut self, ctx: &Bound<'_, PyAny>) -> PyResult<()> {
        if !self.s.untracked_dicts {
            return Ok(());
        }
        let py = ctx.py();
        let armed = self.owner.as_ref().is_some_and(|owner| {
            let referent = owner.reference.bind(py).upgrade();
            !owner.freed() && referent.is_some_and(|referent| referent.is(ctx))
        });
        if !armed {
            self.owner = Owner::new(ctx)?;
        }
        Ok(())
    }

    /// Writes the copies the walker holds into each dict it mirrors that
    /// something besides the walker holds, as a context that lives on, a
    /// copy of it or a caller does, and from then on writes each call's
    /// copies when it ends. `freed` if the walker is being freed by
    /// reference counting, where nothing holds it, and so no context that
    /// the collector frees with it holds the dicts either.
    pub(crate) fn release(&mut self, py: Python<'_>, freed: bool) {
        // A finalizer runs once: a walker it makes reachable again leaves
        // no copies unwritten between calls from then on.
        self.eager = true;
        if !self.unwritten() {
            return;
        }
        let Some((objects, sources)) = &self.mirror else {
            return;
        };
        // See the module doc: only before 3.14 does the walker arm one.
        let dying = !freed && self.owner.as_ref().is_some_and(|owner| !owner.freed());
        let held = |dict: &Py<PyDict>| {
            let dict = dict.bind(py);
            // The walker's own references: the mirror's, and the memo's
            // when the walk captured the context and so the dict.
            let own = match self.memo.get(dict.as_ptr() as usize) {
                Some(entry) if entry.original.bind(py).is(dict) => 2,
                _ => 1,
            };
            refcount(dict) - own - isize::from(dying) > 0
        };
        // A destructor cannot raise; the dicts keep what was written, and a
        // walker that lives on takes the rest from them at its next sync,
        // as copies Python code added, but what it owes `_sources`.
        let written = match (held(objects), held(sources)) {
            (_, true) => self.flush(py, false),
            (true, false) => self.flush_objects(py),
            _ => return,
        };
        if written.is_err() {
            self.memo.clear();
            self.dirty.clear();
            self.truncated.clear();
            self.mirrored = 0;
        }
    }

    /// Writes the copies into `objects` alone, for a `_sources` nothing
    /// else holds: the write goes into a copy of `_sources` that is
    /// dropped, so the dict a dying context holds keeps nothing alive, and
    /// the walker owes `_sources` each write it made there, holding the
    /// originals, as the collector sees, until its next write makes them,
    /// should it live on. For that write, the walker still takes
    /// `_sources` as it last saw it.
    fn flush_objects(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some((objects, sources)) = (self.mirror.as_ref())
            .map(|(objects, sources)| (objects.clone_ref(py), sources.bind(py).clone()))
        else {
            return Ok(());
        };
        let spare = sources.copy()?.unbind();
        let memoized: AddressMap<()> = self.memo.iter().map(|(id, _)| (id, ())).collect();
        let last = self.sources_last.as_ref().map(|last| last.clone_ref(py));
        let seen = (self.sources_mirrored, last, self.sources_only.clone());
        self.mirror = Some((objects.clone_ref(py), spare));
        self.owing = Some(Vec::new());
        let written = self.flush(py, false);
        let owing = self.owing.take().unwrap_or_default();
        self.mirror = Some((objects, sources.clone().unbind()));
        (self.sources_mirrored, self.sources_last, self.sources_only) = seen;
        // The writes it made begin with what it owed, in order: unless a
        // write failed among those, which made none of the rest, it owes
        // what it wrote; otherwise all it owed still.
        if owing.len() >= self.owed.len() {
            self.owed = owing;
        }
        // The copies a rebuild let go of, as Python removed them from
        // `objects`, have originals `_sources` holds without one. The keys
        // are read in order, which, unlike looking an id up, runs no key's
        // own methods.
        let dropped: Vec<_> = (sources.keys().iter())
            .filter(|key| {
                id_of(key).is_some_and(|id| memoized.contains_key(&id) && !self.memo.contains(id))
            })
            .collect();
        self.note_sources(dropped);
        written
    }

    /// `Walker.__traverse__`: every Python object the walker holds.
    pub(crate) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(ctx) = &self.ctx {
            visit.call(ctx)?;
        }
        if let Some(attributes) = &self.attributes {
            visit.call(attributes)?;
        }
        visit.call(&self.string_limit)?;
        visit.call(&self.truncation)?;
        visit.call(&self.namespaces)?;
        visit.call(&self.node_marker)?;
        for (_, entry) in self.memo.iter() {
            if let Slot::Value(captured) = &entry.slot {
                visit.call(captured)?;
            }
            visit.call(&entry.original)?;
        }
        for info in self.types.values() {
            info.traverse(visit)?;
        }
        if let Some((objects, sources)) = &self.mirror {
            visit.call(objects)?;
            visit.call(sources)?;
        }
        if let Some(last) = &self.sources_last {
            visit.call(last)?;
        }
        visit.call(&self.stock)?;
        if let Some(known) = &self.ctx_type {
            known.traverse(visit)?;
        }
        if let Some(owner) = &self.owner {
            visit.call(&owner.reference)?;
        }
        for owed in self.owed.iter().chain(self.owing.iter().flatten()) {
            visit.call(&owed.original)?;
        }
        Ok(())
    }

    /// `Walker.__clear__`, which the collector calls only after the
    /// walker's finalizer had its chance to write.
    pub(crate) fn clear(&mut self) {
        self.memo.clear();
        self.types.clear();
        self.mirror = None;
        self.mirrored = 0;
        self.sources_mirrored = 0;
        self.sources_last = None;
        self.sources_only.clear();
        self.ctx_type = None;
        self.owner = None;
        self.owed.clear();
        self.owing = None;
    }
}
