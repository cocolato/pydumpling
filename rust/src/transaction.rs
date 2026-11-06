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

use pyo3::exceptions::{PyRecursionError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;

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

impl State {
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
        self.ctx = None;
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
                self.memo.commit();
                self.publish_nodes(py)?;
                Ok(captured)
            }
            Err(Halt::Py(err)) => {
                // The Python walk keeps whatever it did before the error.
                self.memo.commit();
                self.publish_nodes(py)?;
                Err(err)
            }
            Err(Halt::Bail) => {
                self.memo.rollback();
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

    fn publish_nodes(&self, py: Python<'_>) -> PyResult<()> {
        self.ctx(py)?.setattr(intern!(py, "_nodes"), self.nodes)
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
        // A capture re-entered from a signal handler may have added copies.
        if self.resync(py, &ctx)? {
            self.flush(py)?;
        }
        ctx.setattr(intern!(py, "_nodes"), self.nodes)?;
        // Without its walker the context captures in Python throughout.
        let walker = ctx.getattr(intern!(py, "_walker"))?;
        ctx.setattr(intern!(py, "_walker"), py.None())?;
        let result = prepend(&ctx, call.args).and_then(|args| call.pure.call(args, call.kwargs));
        let restored = ctx.setattr(intern!(py, "_walker"), walker);
        let synced = self.resync(py, &ctx);
        let nodes = ctx
            .getattr(intern!(py, "_nodes"))
            .and_then(|nodes| nodes.extract());
        restored?;
        synced?;
        self.nodes = nodes?;
        result
    }

    /// Gives the context to the Python walk for good, with every copy the
    /// walker made, and runs `call` there.
    fn retire<'py>(
        &mut self,
        py: Python<'py>,
        call: &Call<'_, 'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let ctx = self.ctx(py)?;
        if self.resync(py, &ctx)? {
            self.flush(py)?;
        }
        self.memo.clear();
        self.mirror = None;
        self.mirrored = 0;
        ctx.setattr(intern!(py, "_walker"), py.None())?;
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
}
