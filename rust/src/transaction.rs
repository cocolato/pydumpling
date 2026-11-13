//! Each call from Python runs as a transaction.
//!
//! The native walk never runs application code. Where the Python walk
//! would, or where its result would depend on its own recursion limit, or
//! the native stack runs short, every change is undone and the call runs in
//! the Python walk instead, from the very same state, so the snapshot is
//! the one the Python walk alone would make and application code runs once.
//! A context whose state only the Python walk reproduces goes to it for
//! good.

use std::ptr::addr_of;
use std::sync::atomic::Ordering::Relaxed;

use pyo3::exceptions::{PyException, PyReferenceError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::arguments::{parse, Parsed};
use crate::dispatch::prepend;
use crate::walker::{Call, Halt, State, Walk};

/// Native stack the walk may use before handing a call back to Python.
const STACK_BUDGET: usize = 256 * 1024;
/// Native stack left untouched for the hand-back itself and the Python
/// helpers called between two depth checks.
const STACK_RESERVE: usize = 64 * 1024;
/// The share of the remaining native stack the walk may use: collections
/// and finalizers that run between two checks need the rest.
const STACK_SHARE: usize = 4;
/// Walk frames (calls of the Python walk's recursive methods) the start
/// check makes room for: about twice what a walk to the default
/// `max_depth` needs.
const CHECK_FROM: u64 = 32;
/// Python frames the Python walk may push beyond its recursive chain, for
/// helpers, generators, and hash methods.
const MARGIN: u64 = 64;
/// The deepest frame a look goes to: walking to it costs a microsecond,
/// and a stack shallower than twice this depth is far from any recursion
/// limit that high.
const LOOK_LIMIT: u64 = 1024;
/// `sys.monitoring` tool ids, 0 to 5.
const MONITORING_TOOLS: u8 = 6;

impl State {
    /// The snapshot context of the call in progress.
    pub(crate) fn ctx<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.ctx {
            Some(ctx) => Ok(ctx.bind(py).clone()),
            None => Err(PyReferenceError::new_err("the walker is not capturing")),
        }
    }

    /// `ctx.objects`, as the call in progress read it.
    pub(crate) fn objects<'a, 'py>(&'a self, py: Python<'py>) -> PyResult<&'a Bound<'py, PyDict>> {
        match &self.objects {
            Some(objects) => Ok(objects.bind(py)),
            None => Err(PyReferenceError::new_err("the walker is not capturing")),
        }
    }

    /// `ctx._sources`, as the call in progress read it.
    pub(crate) fn sources<'a, 'py>(&'a self, py: Python<'py>) -> PyResult<&'a Bound<'py, PyDict>> {
        match &self.sources {
            Some(sources) => Ok(sources.bind(py)),
            None => Err(PyReferenceError::new_err("the walker is not capturing")),
        }
    }

    /// Enters a walk function, handing the call back to Python if the
    /// Python walk could overflow here or the native stack runs short.
    pub(crate) fn enter(&mut self, py: Python<'_>) -> Walk<()> {
        self.frames += 1;
        let here = 0u8;
        if self.origin.abs_diff(addr_of!(here) as usize) > self.stack_budget {
            return Err(Halt::Bail);
        }
        if self.frames > self.room {
            // Twice the room, from one more look: a few at most per call,
            // as the native stack budget ends a deeper walk first.
            let room = self.room * 2;
            if !self.room_for(py, room)? {
                return Err(Halt::Bail);
            }
            self.room = room;
        }
        Ok(())
    }

    /// Leaves a walk function entered with [`State::enter`].
    pub(crate) fn leave(&mut self) {
        self.frames -= 1;
    }

    /// Runs one call from Python as a transaction.
    pub(crate) fn run<'py>(
        &mut self,
        py: Python<'py>,
        ctx: &Bound<'py, PyAny>,
        call: &Call<'_, 'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.ctx = Some(ctx.clone().unbind());
        self.toolless.store(false, Relaxed);
        let result = self.refresh(py).and_then(|usable| {
            if !usable {
                return self.retire(py, call);
            }
            match parse(self.s, call)? {
                Some(parsed) => self.transaction(py, call, &parsed),
                // An argument the walker cannot read without running
                // application code: Python takes the call as it was made.
                None => self.hand_back(py, call),
            }
        });
        (self.ctx, self.attributes) = (None, None);
        (self.objects, self.sources) = (None, None);
        result
    }

    fn transaction<'py>(
        &mut self,
        py: Python<'py>,
        call: &Call<'_, 'py>,
        parsed: &Parsed<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Near its recursion limit the Python walk's result depends on the
        // exact depth, which no cheap look tells: a call the walker cannot
        // show to have room is the Python walk's from the start.
        if !self.room_for(py, CHECK_FROM)? {
            return self.hand_back(py, call);
        }
        self.room = CHECK_FROM;
        let truncation = self.truncation.bind(py).clone();
        let saved = truncation.copy()?;
        let nodes = self.nodes;
        let (objects, sources) = (self.objects(py)?.clone(), self.sources(py)?.clone());
        self.marks = (objects.len(), sources.len());
        self.replaced.clear();
        self.added_namespaces.clear();
        self.frames = 0;
        let origin = 0u8;
        self.origin = addr_of!(origin) as usize;
        self.stack_budget = match stacker::remaining_stack() {
            Some(remaining) => {
                STACK_BUDGET.min(remaining.saturating_sub(STACK_RESERVE) / STACK_SHARE)
            }
            None => STACK_BUDGET,
        };
        let result = match parsed {
            Parsed::Value { value, depth } => self.value(py, value, *depth),
            Parsed::Mapping {
                source,
                depth,
                priority,
                counted,
                container,
            } => self.mapping(py, source, *depth, priority.as_ref(), *counted, container),
        };
        match result {
            Ok(captured) => {
                self.commit(py)?;
                Ok(captured)
            }
            Err(Halt::Py(err)) => {
                // The Python walk keeps whatever it did before the error.
                self.commit(py)?;
                Err(err)
            }
            Err(Halt::Bail) => {
                self.rollback(py, &objects, &sources)?;
                self.nodes = nodes;
                truncation.clear();
                truncation.update(saved.as_mapping())?;
                let namespaces = self.namespaces.bind(py);
                for id in self.added_namespaces.drain(..) {
                    namespaces.discard(id)?;
                }
                self.hand_back(py, call)
            }
        }
    }

    /// Keeps what the call did: its copies are in `objects` already, and
    /// the node count is written to the instance dict as stock attribute
    /// access writes it.
    fn commit(&mut self, py: Python<'_>) -> PyResult<()> {
        self.replaced.clear();
        if let Some(attributes) = &self.attributes {
            attributes
                .bind(py)
                .set_item(intern!(py, "_nodes"), self.nodes)?;
        }
        Ok(())
    }

    /// Takes out of `objects` and `_sources` what the call wrote there, so
    /// the Python walk starts from the dicts as the call found them: each
    /// copy the call replaced goes back where it was, and the entries it
    /// added, which a dict keeps after its older ones, are removed.
    fn rollback(
        &mut self,
        py: Python<'_>,
        objects: &Bound<'_, PyDict>,
        sources: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        for (key, copy) in self.replaced.drain(..).rev() {
            objects.set_item(key, copy)?;
        }
        for (dict, mark) in [(objects, self.marks.0), (sources, self.marks.1)] {
            let added = dict.len().saturating_sub(mark);
            let newest_first = dict.call_method0(intern!(py, "__reversed__"))?;
            let keys: Vec<_> = newest_first
                .try_iter()?
                .take(added)
                .collect::<PyResult<_>>()?;
            for key in keys {
                dict.del_item(key)?;
            }
        }
        Ok(())
    }

    /// Sets `ctx._walker` as stock attribute access does, without the
    /// class's hooks, which the Python walk never runs for it.
    fn set_walker(
        &self,
        py: Python<'_>,
        ctx: &Bound<'_, PyAny>,
        walker: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let name = intern!(py, "_walker");
        self.s.object_setattr.bind(py).call1((ctx, name, walker))?;
        Ok(())
    }

    /// Runs `call` in the Python walk, from the state the call found. The
    /// context holds no walker meanwhile, so it captures in Python
    /// throughout, as does a copy made of it then.
    fn hand_back<'py>(
        &mut self,
        py: Python<'py>,
        call: &Call<'_, 'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.handbacks += 1;
        let ctx = self.ctx(py)?;
        let name = intern!(py, "_walker");
        let walker = self.s.object_getattribute.bind(py).call1((&ctx, name))?;
        self.set_walker(py, &ctx, &py.None().into_bound(py))?;
        let result = prepend(&ctx, call.args).and_then(|args| call.pure.call(args, call.kwargs));
        self.set_walker(py, &ctx, &walker)?;
        result
    }

    /// Gives the context to the Python walk for good, and runs `call`
    /// there.
    fn retire<'py>(
        &mut self,
        py: Python<'py>,
        call: &Call<'_, 'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let ctx = self.ctx(py)?;
        self.set_walker(py, &ctx, &py.None().into_bound(py))?;
        call.pure.call(prepend(&ctx, call.args)?, call.kwargs)
    }
}

impl State {
    /// Whether the caller surely has room for `levels` walk frames, each
    /// with [`MARGIN`] frames to spare, from one look that recurses nowhere.
    ///
    /// `sys._getframe(n)` fails exactly when fewer than `n + 1` frames are
    /// visible, and a visible frame counts at most twice toward the limit
    /// (Python 3.13+ runs `__init__` beneath a hidden shim frame), so its
    /// own failure proves the room. That failure is raised in C before any
    /// audit hook runs, so it is exactly a `ValueError` with no traceback.
    /// A look that succeeds runs the audit hooks, the only Python code a
    /// look runs, and proves nothing, nor does a hook's refusal: either
    /// hands the call back, though an exception that is not an `Exception`,
    /// such as a signal handler's `KeyboardInterrupt`, goes on. A limit too
    /// low for `levels` leaves no room, even on a thread that calls the
    /// walker with no Python frame, where any look fails.
    fn room_for(&self, py: Python<'_>, levels: u64) -> PyResult<bool> {
        let limit: u64 = self.s.recursion_limit.bind(py).call0()?.extract()?;
        let Some(spare) = limit.checked_sub(levels + MARGIN + 1) else {
            return Ok(false);
        };
        let look = (spare / 2).min(LOOK_LIMIT);
        match self.s.get_frame.bind(py).call1((look,)) {
            Ok(_) => Ok(false),
            Err(err)
                if err.get_type(py).is(py.get_type::<PyValueError>())
                    && err.traceback(py).is_none() =>
            {
                Ok(true)
            }
            Err(err) if err.is_instance_of::<PyException>(py) => Ok(false),
            Err(err) => Err(err),
        }
    }

    /// Hands the call back before the walk runs one of pydumpling's own
    /// helpers, if a profiler, tracer or monitoring tool would see it: the
    /// hook is application code, which the Python walk runs at its own
    /// calls instead.
    pub(crate) fn before_python(&self, py: Python<'_>) -> Walk<()> {
        match self.hooked(py)? {
            true => Err(Halt::Bail),
            false => Ok(()),
        }
    }

    /// Whether Python code would run a profile or trace function, or a
    /// `sys.monitoring` tool. Hooks are asked for before each Python call,
    /// since a signal handler, a collection or another thread may set one
    /// while the walk runs; tool ids, once per call.
    fn hooked(&self, py: Python<'_>) -> PyResult<bool> {
        let s = self.s;
        match &s.all_events {
            // Every profile or trace function, one set from C without an
            // object too, and every tool's global events, from any thread.
            Some(all_events) => {
                if all_events.bind(py).call0()?.is_truthy()? {
                    return Ok(true);
                }
            }
            None => {
                if !s.get_profile.bind(py).call0()?.is_none()
                    || !s.get_trace.bind(py).call0()?.is_none()
                {
                    return Ok(true);
                }
            }
        }
        if self.toolless.load(Relaxed) {
            return Ok(false);
        }
        // A tool with only per-code events still has an id in use.
        let get_tool = s.get_tool.bind(py);
        for tool in 0..MONITORING_TOOLS {
            if !get_tool.call1((tool,))?.is_none() {
                return Ok(true);
            }
        }
        self.toolless.store(true, Relaxed);
        Ok(false)
    }
}
