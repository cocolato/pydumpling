//! The native entries of `ValuesMixin.value` and `ValuesMixin._mapping`,
//! and the two ways the walk reaches Python code once it has started.
//!
//! On a context whose class captures natively (`_native_walk`, chosen once
//! per class), each of the two methods calls its entry here first, bound as
//! `_STATICS.value` and `_STATICS.mapping`. An entry checks that the class
//! was judged ([`Statics::judged`]), retiring an instance of one that was
//! not, reads the context's state anew and walks. It returns the copy,
//! raises what the Python walk would raise, or returns `_IN_PYTHON` when it
//! decides, before the call has done anything, that the call is the Python
//! walk's; the method then runs the call itself, in its own frame. Once the
//! walk has done something it never goes back: a node it does not model, or
//! one below its native stack budget, goes to the Python walk in place
//! ([`State::python_value`]), and an operation that may run application
//! code runs once, where the Python walk runs it ([`State::application`]),
//! with the node count synced around both. Other Python code that runs
//! inside a call, such as `FakeContainer.__hash__` or a finalizer, is not
//! synced (`rust/DESIGN.md`, "Hooks and signals"). The walker keeps no memo
//! of its own: like the Python walk, it looks every copy up in
//! `ctx.objects` and writes it there, and its original into `ctx._sources`,
//! as it makes it.

use std::ptr::addr_of;
use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::exceptions::PyAttributeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::state::State;
use crate::statics::{type_address, Statics};
use crate::types::fresh_dict;

/// Why the walk stopped early.
pub(crate) enum Halt {
    /// A Python exception, raised as the Python walk would raise it.
    Py(PyErr),
    /// Found before the call did anything: the context holds state only the
    /// Python walk reproduces, so this call and every later one run there.
    Retire,
    /// Found before the call did anything: this call runs in the Python
    /// walk, and the context stays native.
    InPython,
}

impl From<PyErr> for Halt {
    fn from(err: PyErr) -> Self {
        Halt::Py(err)
    }
}

/// A walk step's result.
pub(crate) type Walk<T> = Result<T, Halt>;

/// Calls and nodes the walker sent to the Python walk, in every thread and
/// for every copy of the package: `_STATICS.fallbacks`.
static FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Depths the walker takes; anything larger runs in Python, whose ints
/// never overflow.
const DEPTH_LIMIT: u64 = 1 << 62;

/// Native stack a call may use below its entry, at most: about 1,420
/// levels of nested lists, 720 of dicts or 590 of objects (macOS arm64).
const STACK_BUDGET: usize = 1536 << 10;
/// Native stack left untouched beneath the walk, for what runs there:
/// application code it runs in place, the Python walk of a subtree it
/// sends there, collections and finalizers, and from 3.14 CPython's own
/// checks of the C stack. A call uses at most half of what is left above
/// it.
const STACK_RESERVE: usize = 128 << 10;
/// The least native stack a call starts with: with less than this, which
/// is 192 KiB left, the call runs in the Python walk from its start.
const STACK_FLOOR: usize = 32 << 10;

/// How many calls and nodes the walker sent to the Python walk.
pub(crate) fn fallbacks() -> u64 {
    FALLBACKS.load(Ordering::Relaxed)
}

/// `_IN_PYTHON`, counted: the method runs the call itself.
pub(crate) fn in_python<'py>(py: Python<'py>, s: &Statics) -> Bound<'py, PyAny> {
    FALLBACKS.fetch_add(1, Ordering::Relaxed);
    s.in_python.bind(py).clone()
}

/// `_IN_PYTHON`, after giving `ctx` to the Python walk for good:
/// `ctx._native_walk = False`.
pub(crate) fn retire<'py>(s: &Statics, ctx: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py();
    ctx.setattr(intern!(py, "_native_walk"), false)?;
    Ok(in_python(py, s))
}

/// [`retire`] for a context whose class was never judged
/// ([`Statics::judged`]), before anything of the class runs: the flag is
/// written into the instance dict, past a `__setattr__` nothing compared
/// with the stock one, which makes that dict exist. Out of line, as no
/// stock context comes here.
#[cold]
#[inline(never)]
fn unjudged<'py>(s: &Statics, ctx: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py();
    fresh_dict(&s.instance_dict, ctx)?.set_item(intern!(py, "_native_walk"), false)?;
    Ok(in_python(py, s))
}

/// An exact int depth the walk can add to without overflow.
pub(crate) fn exact_depth(s: &Statics, depth: &Bound<'_, PyAny>) -> Option<i64> {
    if type_address(depth) != s.int {
        return None;
    }
    depth
        .extract::<i64>()
        .ok()
        .filter(|depth| depth.unsigned_abs() <= DEPTH_LIMIT)
}

/// Whether `priority` is an exact tuple, list, set, or frozenset of exact
/// str names, which a mapping's str keys are compared with running only
/// builtin code.
pub(crate) fn str_items(s: &Statics, priority: &Bound<'_, PyAny>) -> PyResult<bool> {
    if ![s.tuple, s.list, s.set, s.frozenset].contains(&type_address(priority)) {
        return Ok(false);
    }
    // Iterating an exact builtin container runs no code.
    for item in priority.try_iter()? {
        if type_address(&item?) != s.str_ {
            return Ok(false);
        }
    }
    Ok(true)
}

/// How much native stack a call starting here may use, or `None` if it
/// runs in Python.
fn stack_budget() -> Option<usize> {
    let budget = match stacker::remaining_stack() {
        Some(remaining) => STACK_BUDGET.min(remaining.saturating_sub(STACK_RESERVE) / 2),
        None => STACK_BUDGET,
    };
    (budget >= STACK_FLOOR).then_some(budget)
}

/// Runs `walk` on the state of `ctx` natively, or returns `_IN_PYTHON`.
pub(crate) fn call<'py>(
    s: &'static Statics,
    ctx: &Bound<'py, PyAny>,
    walk: impl FnOnce(&mut State) -> Walk<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py();
    // Near the end of the native stack, the call runs in Python from its
    // start, and the context stays native for the next one.
    let Some(stack_budget) = stack_budget() else {
        return Ok(in_python(py, s));
    };
    let Some(mut state) = State::load(py, s, ctx, stack_budget)? else {
        return retire(s, ctx);
    };
    let origin = 0u8;
    state.origin = addr_of!(origin) as usize;
    match walk(&mut state) {
        Ok(captured) => {
            state.sync_out(py)?;
            Ok(captured)
        }
        Err(Halt::Py(err)) => {
            // The Python walk keeps whatever it did before the error.
            state.sync_out(py)?;
            Err(err)
        }
        Err(Halt::Retire) => retire(s, ctx),
        Err(Halt::InPython) => Ok(in_python(py, s)),
    }
}

impl State {
    /// Whether the walk has used up its native stack here.
    pub(crate) fn stack_spent(&self) -> bool {
        let here = 0u8;
        self.origin.abs_diff(addr_of!(here) as usize) > self.stack_budget
    }

    /// Writes the node count to the context's `_nodes`, if it does not hold
    /// it already. Where Python code deleted it, nothing is written: the
    /// Python walk does not make it again.
    pub(crate) fn sync_out(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.written != Some(self.nodes) && !self.missing {
            self.ctx(py).setattr(intern!(py, "_nodes"), self.nodes)?;
            self.written = Some(self.nodes);
        }
        Ok(())
    }

    /// Reads the node count back from the context, after Python code that
    /// may have captured through it. One that is not an exact int from 0 to
    /// 2**64 - 1 is not read: the walk counts on from its own, and writes it
    /// there at the next sync. A deleted one is read again where the Python
    /// walk next reads it ([`State::read_missing`]).
    fn sync_in(&mut self, py: Python<'_>) -> PyResult<()> {
        let found = self.ctx(py).getattr_opt(intern!(py, "_nodes"))?;
        self.missing = found.is_none();
        self.written = found
            .filter(|count| type_address(count) == self.s.int)
            .and_then(|count| count.extract::<u64>().ok());
        if let Some(count) = self.written {
            self.nodes = count;
        }
        Ok(())
    }

    /// `_nodes` read where the Python walk reads it, after Python code
    /// deleted it: the read raises the interpreter's `AttributeError`, as
    /// there, or takes the count something stored since.
    #[cold]
    #[inline(never)]
    pub(crate) fn read_missing(&mut self, py: Python<'_>) -> PyResult<()> {
        self.ctx(py).getattr(intern!(py, "_nodes"))?;
        self.sync_in(py)
    }

    /// Runs `operation`, a Python operation that may run application code,
    /// once, where the Python walk runs it: the node count is written to the
    /// context before and read back after, so code that captures through
    /// the context counts on from the walk's count and the walk from its,
    /// and the facts cached about types, which that code may change, are
    /// dropped.
    pub(crate) fn application<T>(
        &mut self,
        py: Python<'_>,
        operation: impl FnOnce() -> PyResult<T>,
    ) -> Walk<T> {
        self.started = true;
        self.sync_out(py)?;
        let result = operation();
        let synced = self.sync_in(py);
        self.forget_types();
        let result = result?;
        synced?;
        Ok(result)
    }

    /// `ValuesMixin.value(ctx, value, depth)`, run in the Python walk in
    /// place, for a node the walker does not model or one below its stack
    /// budget: `_native_walk` is false on the instance until it returns, so
    /// the node and everything under it run there, with the arguments the
    /// Python walk passes and no frame of the walker's between them. Then
    /// the instance's entry is removed where the flag read `True`, so the
    /// context reads its class's flag again, even where that `True` was the
    /// instance's own; any other value it read is set back.
    pub(crate) fn python_value<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
        depth: i64,
    ) -> Walk<Bound<'py, PyAny>> {
        FALLBACKS.fetch_add(1, Ordering::Relaxed);
        let ctx = self.ctx(py).clone();
        let flag = intern!(py, "_native_walk");
        let previous = ctx.getattr(flag)?;
        let false_ = self.s.false_.bind(py);
        let set = !previous.is(false_);
        if set {
            ctx.setattr(flag, false_)?;
        }
        let function = self.s.value_function.bind(py);
        let result = self.application(py, || function.call1((&ctx, value, depth)));
        let restored = match set {
            false => Ok(()),
            true if previous.is(self.s.true_.bind(py)) => match ctx.delattr(flag) {
                // Code in the subtree removed it already.
                Err(err) if err.is_instance_of::<PyAttributeError>(py) => Ok(()),
                other => other,
            },
            true => ctx.setattr(flag, previous),
        };
        let captured = result?;
        restored?;
        Ok(captured)
    }
}

/// The walker's table for one import of `pydumpling.snapshot.values`,
/// and its native entries.
///
/// `values` builds one when it is imported and binds `value` and `mapping`
/// as `_native_value` and `_native_mapping`. The table lives as long as
/// the process: a reloaded package builds a new one, and the old one keeps
/// the classes its snapshots were made of.
#[pyclass(module = "pydumpling._native", name = "Statics", frozen)]
pub struct StaticsHandle {
    pub(crate) table: &'static Statics,
}

#[pymethods]
impl StaticsHandle {
    /// Reads the table from the `values` module namespace and its
    /// `identity` module.
    #[new]
    fn new(values: &Bound<'_, PyDict>, identity: &Bound<'_, PyAny>) -> PyResult<Self> {
        let table = Statics::load(values, identity)?;
        Ok(Self {
            table: Box::leak(Box::new(table)),
        })
    }

    /// `ValuesMixin.value(ctx, value, depth)` after its memo lookup, its
    /// node check, its count and the exact scalars: the copy, or
    /// `_IN_PYTHON`.
    fn value<'py>(
        &self,
        ctx: &Bound<'py, PyAny>,
        value: &Bound<'py, PyAny>,
        depth: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let s = self.table;
        if !s.judged(ctx)? {
            return unjudged(s, ctx);
        }
        let Some(depth) = exact_depth(s, depth) else {
            // A depth pydumpling's own callers never pass.
            return retire(s, ctx);
        };
        call(s, ctx, |state| {
            // `value` counted the node already; the walk checks and counts
            // it as its own `value` does.
            state.nodes = state.nodes.checked_sub(1).ok_or(Halt::Retire)?;
            state.fresh_value(ctx.py(), value, depth)
        })
    }

    /// `ValuesMixin._mapping(ctx, source, depth, priority=, counted=,
    /// container=)`: the copy, or `_IN_PYTHON`.
    fn mapping<'py>(
        &self,
        ctx: &Bound<'py, PyAny>,
        source: &Bound<'py, PyAny>,
        depth: &Bound<'py, PyAny>,
        priority: &Bound<'py, PyAny>,
        counted: &Bound<'py, PyAny>,
        container: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let s = self.table;
        let py = ctx.py();
        if !s.judged(ctx)? {
            return unjudged(s, ctx);
        }
        let counted = match counted {
            flag if flag.is(s.true_.bind(py)) => Some(true),
            flag if flag.is(s.false_.bind(py)) => Some(false),
            _ => None,
        };
        let container_read =
            container.is(s.dict_cls.bind(py)) || container.is(s.fake_dict.cls.bind(py));
        // Only the shapes pydumpling's own callers pass, read without
        // running any code.
        match (exact_depth(s, depth), counted) {
            (Some(depth), Some(counted)) if container_read && str_items(s, priority)? => {
                call(s, ctx, |state| {
                    state.mapping_entry(py, source, depth, priority, counted, container)
                })
            }
            _ => retire(s, ctx),
        }
    }

    /// How many native calls went to the Python walk, in this process.
    #[getter]
    fn fallbacks(&self) -> u64 {
        fallbacks()
    }

    /// Records `cls`, the composed context class `values._stock` makes
    /// native, which the entries then take as judged without a lookup. The
    /// first class named stays.
    fn stock(&self, cls: Bound<'_, PyAny>) {
        let _ = self.table.stock.set(cls.unbind());
    }
}
