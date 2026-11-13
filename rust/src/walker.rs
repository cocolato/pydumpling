//! The `Walker` Python class and the state it reads from the context.
//!
//! A `SnapshotContext` keeps its walker in `_walker`; [`crate::dispatch`]
//! routes `ctx.value(...)` and `ctx._mapping(...)` to it, and
//! [`crate::transaction`] runs each call. The walker keeps no memo of its
//! own: like the Python walk, it looks every copy up in `ctx.objects` and
//! writes it there, and its original into `ctx._sources`, as it makes it.

use std::sync::atomic::AtomicBool;

use pyo3::exceptions::PyRecursionError;
use pyo3::gc::PyVisit;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, PyString, PyTuple, PyType};
use pyo3::PyTraverseError;

use crate::address::AddressMap;
use crate::shadow::{unshadowed, ContextType};
use crate::statics::{type_address, Statics, StaticsHandle};
use crate::text::{budget, plain_budgets};
use crate::types::TypeInfo;

/// Why the walk stopped early.
pub(crate) enum Halt {
    /// A Python exception, raised as the Python walk would raise it.
    Py(PyErr),
    /// The Python walk would run application code here, or the walker
    /// cannot show it room clear of its recursion limit, or the native
    /// stack runs short, so the whole call must run there instead.
    Bail,
}

impl From<PyErr> for Halt {
    fn from(err: PyErr) -> Self {
        // Only the Python walk has the Python stack an overflow depends on.
        if Python::attach(|py| err.is_instance_of::<PyRecursionError>(py)) {
            Halt::Bail
        } else {
            Halt::Py(err)
        }
    }
}

/// A walk step's result.
pub(crate) type Walk<T> = Result<T, Halt>;

/// The `SnapshotContext` state the Python walk reads and writes, reread
/// from the context at the start of every call.
pub(crate) struct State {
    pub s: &'static Statics,
    /// The context of the call in progress, its instance dict, where the
    /// walker reads and writes its state as stock attribute access does,
    /// and its `objects` and `_sources`: held only during a call.
    pub(crate) ctx: Option<Py<PyAny>>,
    pub(crate) attributes: Option<Py<PyDict>>,
    pub(crate) objects: Option<Py<PyDict>>,
    pub(crate) sources: Option<Py<PyDict>>,
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
    pub nodes: u64,
    /// Facts about the types met in the current call.
    pub types: AddressMap<TypeInfo>,
    /// Python frames the Python walk would have on the stack right now.
    pub(crate) frames: u64,
    /// How many of those the caller is known to have room for.
    pub(crate) room: u64,
    /// Stack address where the current call entered the walker, and how
    /// much native stack the call may use below it.
    pub(crate) origin: usize,
    pub(crate) stack_budget: usize,
    /// Whether this call found no `sys.monitoring` tool id in use when it
    /// first ran Python code (atomic as a Python class must be `Sync`).
    pub(crate) toolless: AtomicBool,
    /// What a hand-back undoes: how many entries `objects` and `_sources`
    /// held when the call began, the entries of `objects` the call
    /// replaced in place, each with the copy it held, and the namespace
    /// ids the call added.
    pub(crate) marks: (usize, usize),
    pub(crate) replaced: Vec<(Py<PyAny>, Py<PyAny>)>,
    pub added_namespaces: Vec<usize>,
    /// Calls handed back to the Python walk so far.
    pub(crate) handbacks: u64,
    /// The walk methods of the stock context class, `_stock_walk`.
    pub(crate) stock: Py<PyTuple>,
    /// What the context's type held when last checked.
    pub(crate) ctx_type: Option<ContextType>,
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

impl State {
    fn new(py: Python<'_>, s: &'static Statics, stock: Py<PyTuple>) -> PyResult<Self> {
        Ok(Self {
            s,
            ctx: None,
            attributes: None,
            objects: None,
            sources: None,
            max_items: None,
            max_nodes: None,
            max_depth: None,
            max_string_bytes: None,
            code_text_limit: None,
            string_limit: py.None(),
            truncation: PyDict::new(py).unbind(),
            namespaces: PySet::empty(py)?.unbind(),
            node_marker: py.None(),
            nodes: 0,
            types: AddressMap::default(),
            frames: 0,
            room: 0,
            origin: 0,
            stack_budget: 0,
            toolless: AtomicBool::new(false),
            marks: (0, 0),
            replaced: Vec::new(),
            added_namespaces: Vec::new(),
            handbacks: 0,
            stock,
            ctx_type: None,
        })
    }

    /// Rereads the context's budgets and state, which Python code may have
    /// changed since the last call.
    ///
    /// `false` if only the Python walk reproduces what this state does: an
    /// options object or budget other than `DumpOptions` validates, a tally,
    /// tally key or memo container whose own methods it would run, a state
    /// attribute missing, a context class the shadow check refuses, or one
    /// dict as `objects` and `_sources`.
    pub(crate) fn refresh<'py>(&mut self, py: Python<'py>) -> PyResult<bool> {
        // Every attribute name is interned: CPython's type cache keeps the
        // names it sees, and a fresh string kept there pins a memory arena.
        let s = self.s;
        let ctx = self.ctx(py)?;
        // First: nothing reads the context before its classes are checked,
        // and then only its instance dict, whose keys are exact strs.
        let stock = self.stock.bind(py).clone();
        let Some(attributes) = unshadowed(py, s, &mut self.ctx_type, &ctx, &stock)? else {
            return Ok(false);
        };
        let get = |name: &Bound<'py, PyString>| attributes.get_item(name);
        let (
            Some(options),
            Some(code_limit),
            Some(unlimited),
            Some(truncation),
            Some(namespaces),
            Some(objects),
            Some(sources),
            Some(count),
            Some(node_marker),
            Some(layouts),
        ) = (
            get(intern!(py, "options"))?,
            get(intern!(py, "_code_text_limit"))?,
            get(intern!(py, "_unlimited"))?,
            get(intern!(py, "_truncation"))?,
            get(intern!(py, "_namespaces"))?,
            get(intern!(py, "objects"))?,
            get(intern!(py, "_sources"))?,
            get(intern!(py, "_nodes"))?,
            get(intern!(py, "_node_marker"))?,
            get(intern!(py, "_layouts"))?,
        )
        else {
            // Raised by the Python walk, if and where it reads it.
            return Ok(false);
        };
        if type_address(&options) != s.dump_options {
            return Ok(false);
        }
        let Some([items, nodes, depth, string_limit]) = plain_budgets(py, s, &options)? else {
            return Ok(false);
        };
        let budgets = (
            budget(s, &items),
            budget(s, &nodes),
            budget(s, &depth),
            budget(s, &string_limit),
            budget(s, &code_limit),
        );
        let (Some(items), Some(nodes), Some(depth), Some(string), Some(code)) = budgets else {
            return Ok(false);
        };
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
            // `_layout` looks types up in it with its own `get`.
            || type_address(&layouts) != s.dict
            // The Python walk writes every original over its own copy.
            || objects.is(&sources)
        {
            return Ok(false);
        }
        (self.max_items, self.max_nodes, self.max_depth) = (items, nodes, depth);
        (self.max_string_bytes, self.code_text_limit) = (string, code);
        self.string_limit = string_limit.unbind();
        let truncation = truncation.cast_into::<PyDict>()?;
        // Reading a tally, as before a helper the Python walk may not
        // count with, may run no key's `__eq__`.
        if !s.str_keys(&truncation) {
            return Ok(false);
        }
        self.truncation = truncation.unbind();
        self.namespaces = namespaces.cast_into::<PySet>()?.unbind();
        self.node_marker = node_marker.unbind();
        self.nodes = count;
        self.attributes = Some(attributes.unbind());
        self.objects = Some(objects.unbind());
        self.sources = Some(sources.unbind());
        // A class may have changed since the last call. Within a call only
        // code the walk lets run (a signal handler, a collection, another
        // thread) can change one, and the next call sees it.
        self.types.clear();
        Ok(true)
    }
}

/// The native walk behind a `SnapshotContext`: `Walker(statics, stock)`
/// captures for the context that keeps it in `_walker`, whose `value` and
/// `_mapping` calls [`crate::dispatch::Dispatch`] sends to it. `statics` is
/// the `values` module's table, `stock` the walk methods it mirrors. The
/// walker reads the context it captures for anew at every call and holds
/// none of its copies, so a shallow copy of a context shares its walker.
#[pyclass(module = "pydumpling._native")]
pub struct Walker {
    pub(crate) state: State,
}

#[pymethods]
impl Walker {
    #[new]
    fn new(
        py: Python<'_>,
        statics: PyRef<'_, StaticsHandle>,
        stock: Bound<'_, PyTuple>,
    ) -> PyResult<Self> {
        Ok(Self {
            state: State::new(py, statics.table, stock.unbind())?,
        })
    }

    /// How many calls went to the Python walk, because it would have run
    /// application code, or the walker could not show it room clear of its
    /// recursion limit.
    #[getter]
    fn handbacks(&self) -> u64 {
        self.state.handbacks
    }

    /// Pickles and deep-copies as `None`, which a context that captures in
    /// Python holds. It reads no state, so it borrows none, and works
    /// during the walker's own call too.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> (Bound<'py, PyType>, Bound<'py, PyTuple>) {
        let py = slf.py();
        (py.None().into_bound(py).get_type(), PyTuple::empty(py))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        let state = &self.state;
        if let Some(ctx) = &state.ctx {
            visit.call(ctx)?;
        }
        for dict in [&state.attributes, &state.objects, &state.sources]
            .into_iter()
            .flatten()
        {
            visit.call(dict)?;
        }
        visit.call(&state.string_limit)?;
        visit.call(&state.truncation)?;
        visit.call(&state.namespaces)?;
        visit.call(&state.node_marker)?;
        for info in state.types.values() {
            info.traverse(&visit)?;
        }
        for (key, copy) in &state.replaced {
            visit.call(key)?;
            visit.call(copy)?;
        }
        visit.call(&state.stock)?;
        if let Some(known) = &state.ctx_type {
            known.traverse(&visit)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        let state = &mut self.state;
        (state.ctx, state.attributes) = (None, None);
        (state.objects, state.sources) = (None, None);
        state.types.clear();
        state.replaced.clear();
        state.ctx_type = None;
    }
}
