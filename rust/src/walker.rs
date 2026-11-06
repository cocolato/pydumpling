//! The `Walker` Python class and the state it mirrors from the context.
//!
//! A `SnapshotContext` keeps its walker in `_walker`; [`crate::dispatch`]
//! routes `ctx.value(...)` and `ctx._mapping(...)` to it,
//! [`crate::transaction`] runs each call, and [`crate::mirror`] keeps the
//! walker's memo and the context's `objects` in step.

use pyo3::exceptions::{PyRecursionError, PyReferenceError};
use pyo3::gc::PyVisit;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, PyString, PyTuple, PyType};
use pyo3::PyTraverseError;

use crate::memo::{AddressMap, Memo};
use crate::statics::{type_address, Statics, StaticsHandle};
use crate::types::TypeInfo;

/// Why the walk stopped early.
pub(crate) enum Halt {
    /// A Python exception, raised as the Python walk would raise it.
    Py(PyErr),
    /// The Python walk would run application code or near its recursion
    /// limit here, so the whole call must run there instead.
    Bail,
}

impl From<PyErr> for Halt {
    fn from(err: PyErr) -> Self {
        // A recursion overflow depends on the Python stack, which the native
        // walk does not reproduce; only the Python walk can say what happens.
        if Python::attach(|py| err.is_instance_of::<PyRecursionError>(py)) {
            Halt::Bail
        } else {
            Halt::Py(err)
        }
    }
}

/// A walk step's result.
pub(crate) type Walk<T> = Result<T, Halt>;

/// A memoized copy: a value, or an exact tuple or frozenset still being
/// filled, whose placeholder is only built if something refers back to it.
pub(crate) enum Slot {
    Value(Py<PyAny>),
    Recursive,
}

/// A memo entry; the original is kept alive so its address stays unique.
pub(crate) struct Entry {
    pub slot: Slot,
    pub original: Py<PyAny>,
}

/// Mirror of the `SnapshotContext` state the Python walk reads and writes,
/// reread from the context at the start of every call.
pub(crate) struct State {
    pub s: &'static Statics,
    /// The context of the call in progress. The walker holds it only during
    /// a call, so a context and its walker never form a reference cycle.
    pub(crate) ctx: Option<Py<PyAny>>,
    pub max_items: Option<u64>,
    pub max_nodes: Option<u64>,
    pub max_depth: Option<u64>,
    pub max_string_bytes: Option<u64>,
    pub code_text_limit: Option<u64>,
    /// `max_string_bytes` as the options hold it, for Python helpers.
    pub string_limit: Py<PyAny>,
    pub truncation: Py<PyDict>,
    pub namespaces: Py<PySet>,
    pub node_marker: Py<PyAny>,
    pub memo: Memo<Entry>,
    pub nodes: u64,
    /// Facts about the types met in the current call.
    pub types: AddressMap<TypeInfo>,
    /// Python frames the Python walk would have on the stack right now.
    pub(crate) frames: u64,
    /// Python frames the caller is known to have room for beyond its own.
    pub(crate) headroom: u64,
    /// Stack address where the current call entered the walker, and how
    /// much native stack the call may use below it.
    pub(crate) origin: usize,
    pub(crate) stack_budget: usize,
    /// Namespace ids added in the current call, undone on a hand-back.
    pub added_namespaces: Vec<usize>,
    /// Calls handed back to the Python walk so far.
    pub(crate) handbacks: u64,
    /// `ctx.objects` and `ctx._sources` as of the last sync.
    pub(crate) mirror: Option<(Py<PyDict>, Py<PyDict>)>,
    /// How many memo entries, oldest first, `ctx.objects` holds.
    pub(crate) mirrored: usize,
    /// Keys of mirrored entries the walk replaced in place since.
    pub(crate) dirty: Vec<usize>,
    /// Keys whose copy is a placeholder the Python walk may replace in
    /// place: a mapping omitted by depth that a later call captures.
    pub(crate) truncated: Vec<usize>,
    /// The context's type and its instance dict getter, if it has one.
    pub(crate) ctx_type: Option<(Py<PyType>, Option<Py<PyAny>>)>,
    /// The size of the context's instance dict when it was last found to
    /// set no walk method.
    pub(crate) instance_len: usize,
}

/// A call from the Python walk's callers, as they made it.
pub(crate) struct Call<'a, 'py> {
    /// The Python walk's function, which takes the call on a hand-back.
    pub pure: &'a Bound<'py, PyAny>,
    /// Whether this is `_mapping` rather than `value`.
    pub mapping: bool,
    pub args: &'a Bound<'py, PyTuple>,
    pub kwargs: Option<&'a Bound<'py, PyDict>>,
}

/// The longest type name of a placeholder the walker builds lazily:
/// `frozenset`, as an exact tuple's or frozenset's.
const PLACEHOLDER_NAME: u64 = "frozenset".len() as u64;

/// Reads a budget as `DumpOptions` validates it: `Some(None)` for `None`,
/// `Some(Some(limit))` for a positive exact int, saturated to `u64`, and
/// `None` for anything the Python walk alone compares as it would.
fn budget(s: &Statics, value: &Bound<'_, PyAny>) -> Option<Option<u64>> {
    if value.is_none() {
        return Some(None);
    }
    if type_address(value) != s.int {
        return None;
    }
    match value.extract::<u64>() {
        Ok(0) => None,
        Ok(limit) => Some(Some(limit)),
        // Too big for u64, or negative.
        Err(_) => match value.gt(0) {
            Ok(true) => Some(Some(u64::MAX)),
            _ => None,
        },
    }
}

impl State {
    fn new(py: Python<'_>, s: &'static Statics) -> PyResult<Self> {
        Ok(Self {
            s,
            ctx: None,
            max_items: None,
            max_nodes: None,
            max_depth: None,
            max_string_bytes: None,
            code_text_limit: None,
            string_limit: py.None(),
            truncation: PyDict::new(py).unbind(),
            namespaces: PySet::empty(py)?.unbind(),
            node_marker: py.None(),
            memo: Memo::new(),
            nodes: 0,
            types: AddressMap::default(),
            frames: 0,
            headroom: 0,
            origin: 0,
            stack_budget: 0,
            added_namespaces: Vec::new(),
            handbacks: 0,
            mirror: None,
            mirrored: 0,
            dirty: Vec::new(),
            truncated: Vec::new(),
            ctx_type: None,
            instance_len: usize::MAX,
        })
    }

    /// Rereads the context's budgets and state, which Python code may have
    /// changed or replaced since the last call, and syncs the memo with
    /// `ctx.objects`.
    ///
    /// `false` if only the Python walk reproduces what this state does: an
    /// options object or budget other than `DumpOptions` validates, a tally
    /// or memo container whose own methods it would run, or a walk method
    /// set on the context itself.
    pub(crate) fn refresh(&mut self, py: Python<'_>) -> PyResult<bool> {
        // Every attribute name is interned: CPython's type cache keeps the
        // names it sees, and a fresh string kept there pins a memory arena.
        let s = self.s;
        let ctx = self.ctx(py)?;
        let options = ctx.getattr(intern!(py, "options"))?;
        if type_address(&options) != s.dump_options {
            return Ok(false);
        }
        let string_limit = options.getattr(intern!(py, "max_string_bytes"))?;
        let budgets = (
            budget(s, &options.getattr(intern!(py, "max_items"))?),
            budget(s, &options.getattr(intern!(py, "max_nodes"))?),
            budget(s, &options.getattr(intern!(py, "max_depth"))?),
            budget(s, &string_limit),
            budget(s, &ctx.getattr(intern!(py, "_code_text_limit"))?),
        );
        let (Some(items), Some(nodes), Some(depth), Some(string), Some(code)) = budgets else {
            return Ok(false);
        };
        if code.is_some_and(|limit| limit < PLACEHOLDER_NAME) {
            // The Python walk names, and so counts truncating, every
            // placeholder it builds; the walker builds them only when used.
            return Ok(false);
        }
        let unlimited = ctx.getattr(intern!(py, "_unlimited"))?;
        let truncation = ctx.getattr(intern!(py, "_truncation"))?;
        let namespaces = ctx.getattr(intern!(py, "_namespaces"))?;
        let objects = ctx.getattr(intern!(py, "objects"))?;
        let sources = ctx.getattr(intern!(py, "_sources"))?;
        let count = ctx.getattr(intern!(py, "_nodes"))?;
        let count = match type_address(&count) == s.int {
            true => count.extract::<u64>().ok(),
            false => None,
        };
        let (Some(count), Ok(objects), Ok(sources)) = (
            count,
            objects.cast_into_exact::<PyDict>(),
            sources.cast_into_exact::<PyDict>(),
        ) else {
            return Ok(false);
        };
        if !unlimited.is(s.false_.bind(py))
            || type_address(&truncation) != s.counter
            || type_address(&namespaces) != s.set
            || !self.unshadowed(py, &ctx)?
        {
            return Ok(false);
        }
        (self.max_items, self.max_nodes, self.max_depth) = (items, nodes, depth);
        (self.max_string_bytes, self.code_text_limit) = (string, code);
        self.string_limit = string_limit.unbind();
        self.truncation = truncation.cast_into::<PyDict>()?.unbind();
        self.namespaces = namespaces.cast_into::<PySet>()?.unbind();
        self.node_marker = ctx.getattr(intern!(py, "_node_marker"))?.unbind();
        self.nodes = count;
        // A class may have changed since the last call; within a call the
        // walk runs no application code, so it cannot.
        self.types.clear();
        self.sync(py, objects, sources, false)?;
        Ok(true)
    }

    /// Whether the context leaves every walk method to its class: an
    /// instance attribute of that name is what the Python walk would call.
    /// Its instance dict is rescanned only when its size changed.
    fn unshadowed(&mut self, py: Python<'_>, ctx: &Bound<'_, PyAny>) -> PyResult<bool> {
        let ty = ctx.get_type();
        if !self
            .ctx_type
            .as_ref()
            .is_some_and(|(known, _)| known.bind(py).is(&ty))
        {
            let info = TypeInfo::compute(py, self.s, &ty)?;
            let getter = info.dict_get.filter(|_| !info.hostile);
            self.ctx_type = Some((ty.clone().unbind(), getter));
            self.instance_len = usize::MAX;
        }
        let Some((_, Some(getter))) = &self.ctx_type else {
            return Ok(false);
        };
        let attributes = getter.bind(py).call1((ctx, &ty))?;
        let Ok(attributes) = attributes.cast_exact::<PyDict>() else {
            return Ok(false);
        };
        if attributes.len() == self.instance_len {
            return Ok(true);
        }
        for key in attributes.keys() {
            // Comparing any other key may run its own `__eq__`.
            if type_address(&key) != self.s.str_ {
                return Ok(false);
            }
            let key = key.cast::<PyString>().map_err(PyErr::from)?;
            // A lone surrogate never spells a method name.
            if let Ok(name) = key.to_str() {
                if self.s.walk_names.iter().any(|walk| walk == name) {
                    return Ok(false);
                }
            }
        }
        self.instance_len = attributes.len();
        Ok(true)
    }

    /// The snapshot context of the call in progress.
    pub(crate) fn ctx<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.ctx {
            Some(ctx) => Ok(ctx.bind(py).clone()),
            None => Err(PyReferenceError::new_err("the walker is not capturing")),
        }
    }
}

/// The native walk behind a `SnapshotContext`.
///
/// `Walker(statics)` captures for the context that keeps it in `_walker`;
/// [`crate::dispatch::Dispatch`] sends the context's `value` and `_mapping`
/// calls to it. `statics` is the table of the `values` module that made it.
/// The walker holds no reference to its context between calls.
#[pyclass(module = "pydumpling._native")]
pub struct Walker {
    pub(crate) state: State,
}

#[pymethods]
impl Walker {
    #[new]
    fn new(py: Python<'_>, statics: PyRef<'_, StaticsHandle>) -> PyResult<Self> {
        Ok(Self {
            state: State::new(py, statics.table)?,
        })
    }

    /// How many calls went to the Python walk, because it would have run
    /// application code or neared its recursion limit.
    #[getter]
    fn handbacks(&self) -> u64 {
        self.state.handbacks
    }

    /// Writes every copy the walker memoized into `ctx.objects` and
    /// `ctx._sources`, where the Python walk keeps them, as before the
    /// context is copied. Called from within a capture of this walker, as
    /// from a signal handler, it leaves them as they are.
    fn flush(slf: &Bound<'_, Self>, ctx: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let Ok(mut walker) = slf.try_borrow_mut() else {
            return Ok(());
        };
        let state = &mut walker.state;
        state.ctx = Some(ctx.clone().unbind());
        let flushed = state.resync(py, ctx).and_then(|synced| match synced {
            true => state.flush(py),
            false => Ok(()),
        });
        state.ctx = None;
        flushed
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        let state = &self.state;
        if let Some(ctx) = &state.ctx {
            visit.call(ctx)?;
        }
        visit.call(&state.string_limit)?;
        visit.call(&state.truncation)?;
        visit.call(&state.namespaces)?;
        visit.call(&state.node_marker)?;
        for (_, entry) in state.memo.iter() {
            if let Slot::Value(captured) = &entry.slot {
                visit.call(captured)?;
            }
            visit.call(&entry.original)?;
        }
        for info in state.types.values() {
            info.traverse(&visit)?;
        }
        if let Some((objects, sources)) = &state.mirror {
            visit.call(objects)?;
            visit.call(sources)?;
        }
        if let Some((ty, getter)) = &state.ctx_type {
            visit.call(ty)?;
            if let Some(getter) = getter {
                visit.call(getter)?;
            }
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        let state = &mut self.state;
        state.memo.clear();
        state.types.clear();
        state.mirror = None;
        state.mirrored = 0;
        state.ctx_type = None;
    }
}
