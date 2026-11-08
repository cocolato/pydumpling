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
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

use pyo3::exceptions::{PyRecursionError, PyReferenceError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyType};

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
/// Walk frames below which the Python recursion limit cannot be near.
const CHECK_FROM: u64 = 32;
/// Python frames the Python walk may push beyond its recursive chain, for
/// helpers, generators, and hash methods.
const MARGIN: u64 = 64;
/// Frames a probe asks for beyond what the walk needs now, at least.
const PROBE_STEP: u64 = 256;
/// The deepest frame the start check looks for: walking to it costs a
/// microsecond, and a stack shallower than twice this depth is far from
/// any recursion limit that high.
const LOOK_LIMIT: u64 = 1024;
/// `sys.monitoring` tool ids, 0 to 5.
const MONITORING_TOOLS: u8 = 6;

/// What `ctx._walker` holds while the Python walk takes a call from the
/// walker: the context then captures in Python, as with `None`, and a copy
/// of it, which shares the dicts the walker wrote before the call, makes
/// the walker write each later call's copies at once.
#[pyclass(module = "pydumpling._native", frozen)]
#[derive(Default)]
pub struct HandBack {
    copied: AtomicBool,
}

#[pymethods]
impl HandBack {
    /// Notes a copy of the context: the walker's copies are written.
    fn flush(&self) {
        self.copied.store(true, Relaxed);
    }

    /// Pickles and copies as `None`, as the walker does.
    fn __reduce__<'py>(&self, py: Python<'py>) -> (Bound<'py, PyType>, Bound<'py, PyTuple>) {
        self.flush();
        (py.None().into_bound(py).get_type(), PyTuple::empty(py))
    }
}

impl State {
    /// The snapshot context of the call in progress.
    pub(crate) fn ctx<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.ctx {
            Some(ctx) => Ok(ctx.bind(py).clone()),
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
        let need = self.frames + MARGIN;
        if need > self.headroom {
            self.before_python(py)?;
            self.headroom = self.probe(py, need)?;
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
        let result = self
            .arm(ctx)
            .and_then(|()| self.refresh(py))
            .and_then(|usable| {
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
        self.ctx = None;
        self.attributes = None;
        result
    }

    fn transaction<'py>(
        &mut self,
        py: Python<'py>,
        call: &Call<'_, 'py>,
        parsed: &Parsed<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let truncation = self.truncation.bind(py).clone();
        let saved = truncation.copy()?;
        let nodes = self.nodes;
        let dirty = self.dirty.len();
        self.memo.begin();
        self.added_namespaces.clear();
        self.frames = 0;
        self.headroom = self.start_headroom(py)?;
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
                self.memo.rollback();
                self.dirty.truncate(dirty);
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

    /// Keeps what the call did: its copies, and the node count, written to
    /// the instance dict as stock attribute access writes it.
    fn commit(&mut self, py: Python<'_>) -> PyResult<()> {
        self.memo.commit();
        if let Some(attributes) = &self.attributes {
            attributes
                .bind(py)
                .set_item(intern!(py, "_nodes"), self.nodes)?;
        }
        if self.eager {
            // Python code the walk ran, as a collection's callbacks, may have
            // changed the dicts since the call's sync checked them.
            self.flush(py, false)?;
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

    /// Runs `call` in the Python walk from the walker's state, then adopts
    /// the state the Python walk leaves, whether it returned or raised.
    fn hand_back<'py>(
        &mut self,
        py: Python<'py>,
        call: &Call<'_, 'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.handbacks += 1;
        let ctx = self.ctx(py)?;
        // The Python walk must see every copy the walker holds, in the dicts
        // this call synced with. The node count needs no write:
        // `ctx._nodes` already holds it, as a transaction publishes it on
        // commit and a rollback restores it.
        self.flush(py, false)?;
        // Without its walker the context captures in Python throughout.
        let name = intern!(py, "_walker");
        let walker = self.s.object_getattribute.bind(py).call1((&ctx, name))?;
        let lent = Bound::new(py, HandBack::default())?;
        self.set_walker(py, &ctx, lent.as_any())?;
        let result = prepend(&ctx, call.args).and_then(|args| call.pure.call(args, call.kwargs));
        let restored = self.set_walker(py, &ctx, &walker);
        if lent.get().copied.load(Relaxed) {
            // A copy shares the dicts from now on.
            self.eager = true;
        }
        let synced = self.resync(py, &ctx);
        restored?;
        synced?;
        result
    }

    /// Gives the context to the Python walk for good, with every copy the
    /// walker made where the Python walk would have put it, in the dicts
    /// last synced with, and runs `call` there.
    fn retire<'py>(
        &mut self,
        py: Python<'py>,
        call: &Call<'_, 'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let ctx = self.ctx(py)?;
        self.flush(py, false)?;
        self.memo.clear();
        self.mirror = None;
        self.mirrored = 0;
        self.sources_mirrored = 0;
        self.sources_last = None;
        self.set_walker(py, &ctx, &py.None().into_bound(py))?;
        call.pure.call(prepend(&ctx, call.args)?, call.kwargs)
    }
}

impl State {
    /// Python frames the caller surely has room for, from one cheap look:
    /// [`CHECK_FROM`] walk frames' worth unless the stack is deep enough
    /// for the limit to be near, which [`State::probe`] then measures.
    ///
    /// `sys._getframe(n)` fails exactly when fewer than `n + 1` frames are
    /// visible, and a visible frame counts at most twice toward the limit
    /// (Python 3.13+ runs `__init__` beneath a hidden shim frame), so one
    /// failed call answers for every call far from the limit, and raises
    /// no audit event. If the stack is deeper than the look goes, or the
    /// look fails, as when an audit hook refuses it, the probe measures.
    fn start_headroom(&self, py: Python<'_>) -> PyResult<u64> {
        let limit: u64 = self.s.recursion_limit.bind(py).call0()?.extract()?;
        let probe = (limit.saturating_sub(CHECK_FROM + MARGIN + 1) / 2).min(LOOK_LIMIT);
        Ok(match self.s.get_frame.bind(py).call1((probe,)) {
            Err(err) if err.is_instance_of::<PyValueError>(py) => CHECK_FROM + MARGIN,
            _ => 0,
        })
    }

    /// How many more Python frames the caller has room for, at least
    /// `need`, measured by recursing into `values._headroom`. The
    /// interpreter does the counting, so every frame the limit counts is
    /// included, visible or not. Hands the call back if `need` frames do
    /// not fit.
    fn probe(&self, py: Python<'_>, need: u64) -> Walk<u64> {
        let fits = |frames: u64| -> Walk<bool> {
            match self.s.headroom.bind(py).call1((frames,)) {
                Ok(_) => Ok(true),
                Err(err) if err.is_instance_of::<PyRecursionError>(py) => Ok(false),
                Err(err) => Err(Halt::Py(err)),
            }
        };
        let ample = need.saturating_add(need.max(PROBE_STEP));
        if fits(ample)? {
            return Ok(ample);
        }
        if !fits(need)? {
            return Err(Halt::Bail);
        }
        let (mut low, mut high) = (need, ample);
        while high - low > 1 {
            let middle = low + (high - low) / 2;
            if fits(middle)? {
                low = middle;
            } else {
                high = middle;
            }
        }
        Ok(low)
    }

    /// Hands the call back before the walk runs Python code, the stack
    /// probe or one of pydumpling's own helpers, if a profiler, tracer or
    /// monitoring tool would see it: the hook is application code, which
    /// the Python walk runs at its own calls instead.
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
