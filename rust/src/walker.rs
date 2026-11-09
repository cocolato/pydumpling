//! The `Walker` Python class and the state it mirrors from the context.
//!
//! A `SnapshotContext` keeps its walker in `_walker`; [`crate::dispatch`]
//! routes `ctx.value(...)` and `ctx._mapping(...)` to it,
//! [`crate::transaction`] runs each call, and [`crate::mirror`] keeps the
//! walker's memo and the context's `objects` in step.

use std::sync::atomic::AtomicBool;

use pyo3::exceptions::PyRecursionError;
use pyo3::gc::PyVisit;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, PyString, PyTuple, PyType};
use pyo3::PyTraverseError;

use crate::flush::Owed;
use crate::memo::{AddressMap, Memo};
use crate::release::{refcount, Owner};
use crate::shadow::{unshadowed, ContextType};
use crate::statics::{type_address, Statics, StaticsHandle};
use crate::text::budget;
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
    /// The context of the call in progress, which the walker holds only
    /// during a call. Originals the memo keeps can still hold the context,
    /// as a captured object that refers back to it does: see
    /// [`crate::release`].
    pub(crate) ctx: Option<Py<PyAny>>,
    /// The context's instance dict during a call, where the walker reads
    /// and writes its state as stock attribute access does.
    pub(crate) attributes: Option<Py<PyDict>>,
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
    /// Whether this call found no `sys.monitoring` tool id in use when it
    /// first ran Python code (atomic as a Python class must be `Sync`).
    pub(crate) toolless: AtomicBool,
    /// Namespace ids added in the current call, undone on a hand-back.
    pub added_namespaces: Vec<usize>,
    /// Calls handed back to the Python walk so far.
    pub(crate) handbacks: u64,
    /// `ctx.objects` and `ctx._sources` as of the last sync.
    pub(crate) mirror: Option<(Py<PyDict>, Py<PyDict>)>,
    /// How many memo entries, oldest first, `ctx.objects` holds.
    pub(crate) mirrored: usize,
    /// How many entries `ctx._sources` held at the last sync or write, and
    /// the newest: Python code added any after it.
    pub(crate) sources_mirrored: usize,
    pub(crate) sources_last: Option<Py<PyAny>>,
    /// Whether each call writes its copies to `ctx.objects` when it ends,
    /// as once a copy of the context may share that dict.
    pub(crate) eager: bool,
    /// Keys of mirrored entries the walk replaced in place since.
    pub(crate) dirty: Vec<usize>,
    /// Whether `ctx.objects` holds a key that is not an id, or `_sources`
    /// one not of a plain kind: only the Python walk looks them up as it would.
    pub(crate) foreign: bool,
    /// Keys whose copy is a placeholder the Python walk may replace in
    /// place: a mapping omitted by depth that a later call captures.
    pub(crate) truncated: Vec<usize>,
    /// The walk methods of the stock context class, `_stock_walk`.
    pub(crate) stock: Py<PyTuple>,
    /// What the context's type held when last checked.
    pub(crate) ctx_type: Option<ContextType>,
    /// The context of the walker's latest call, where a dict the collector
    /// leaves untracked needs it: see [`crate::release`].
    pub(crate) owner: Option<Owner>,
    /// The writes into `ctx._sources` a release that wrote only
    /// `ctx.objects` left out, in order, which the next write makes first:
    /// see [`crate::release`].
    pub(crate) owed: Vec<Owed>,
    /// While such a release writes, the writes it so leaves out.
    pub(crate) owing: Option<Vec<Owed>>,
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

impl State {
    fn new(py: Python<'_>, s: &'static Statics, stock: Py<PyTuple>) -> PyResult<Self> {
        Ok(Self {
            s,
            ctx: None,
            attributes: None,
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
            toolless: AtomicBool::new(false),
            added_namespaces: Vec::new(),
            handbacks: 0,
            mirror: None,
            mirrored: 0,
            sources_mirrored: 0,
            sources_last: None,
            eager: false,
            dirty: Vec::new(),
            foreign: false,
            truncated: Vec::new(),
            stock,
            ctx_type: None,
            owner: None,
            owed: Vec::new(),
            owing: None,
        })
    }

    /// Rereads the context's budgets and state, which Python code may have
    /// changed since the last call, and syncs the memo with `ctx.objects`.
    ///
    /// `false` if only the Python walk reproduces what this state does: an
    /// options object or budget other than `DumpOptions` validates, a tally,
    /// tally key or memo container whose own methods it would run, a memo
    /// key that is not an id, a state attribute missing, a context class
    /// the shadow check refuses, or one dict as `objects` and `_sources`.
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
        let string_limit = options.getattr(intern!(py, "max_string_bytes"))?;
        let budgets = (
            budget(s, &options.getattr(intern!(py, "max_items"))?),
            budget(s, &options.getattr(intern!(py, "max_nodes"))?),
            budget(s, &options.getattr(intern!(py, "max_depth"))?),
            budget(s, &string_limit),
            budget(s, &code_limit),
        );
        let (Some(items), Some(nodes), Some(depth), Some(string), Some(code)) = budgets else {
            return Ok(false);
        };
        if code.is_some_and(|limit| limit < PLACEHOLDER_NAME) {
            // The Python walk names, and so counts truncating, every
            // placeholder it builds; the walker builds them only when used.
            return Ok(false);
        }
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
        // Reading a tally, as before a helper, may run no key's `__eq__`.
        if !s.plain_dict(&truncation) {
            return Ok(false);
        }
        self.truncation = truncation.unbind();
        self.namespaces = namespaces.cast_into::<PySet>()?.unbind();
        self.node_marker = node_marker.unbind();
        self.nodes = count;
        self.attributes = Some(attributes.unbind());
        // A class may have changed since the last call. Within a call only
        // code the walk lets run (a signal handler, a collection, another
        // thread) can change one, and the next call sees it.
        self.types.clear();
        // A copy of the context shares these dicts, and its Python walk
        // may replace a depth placeholder in place, as a hand-back may.
        self.sync(py, objects, sources, self.eager)?;
        Ok(!self.foreign)
    }
}

/// The native walk behind a `SnapshotContext`: `Walker(statics, stock)`
/// captures for the context that keeps it in `_walker`, whose `value` and
/// `_mapping` calls [`crate::dispatch::Dispatch`] sends to it. `statics` is
/// the `values` module's table, `stock` the walk methods it mirrors.
/// Between calls the walker refers to its context only through what its
/// memo keeps, or what it owes `_sources`, the context itself once a walk
/// captured it, and, where its release needs one (see `release.rs`), a
/// weak reference. `values` makes walkers of a subclass whose `__del__` is
/// `Walker.release`; one of this class itself has no finalizer, and so
/// writes nothing as it is freed.
#[pyclass(module = "pydumpling._native", subclass)]
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
    /// application code or neared its recursion limit.
    #[getter]
    fn handbacks(&self) -> u64 {
        self.state.handbacks
    }

    /// Writes every copy the walker memoized into the dicts it last synced
    /// with, as before the context is copied; from then on each call writes
    /// its copies when it ends, so a shallow copy, which shares those dicts,
    /// sees them. A copy made during a native call raises `RuntimeError`.
    fn flush(slf: &Bound<'_, Self>) -> PyResult<()> {
        let Ok(mut walker) = slf.try_borrow_mut() else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "this SnapshotContext is already capturing: cannot copy a native transaction",
            ));
        };
        let state = &mut walker.state;
        state.eager = true;
        state.flush(slf.py(), false)
    }

    /// Writes the walker's copies into a dict it mirrors that outlives it:
    /// see [`crate::release`]. `values` makes it its walkers' `__del__`,
    /// which then pushes no Python frame: the Python recursion limit, a
    /// signal or a tracer cannot stop it, though a spent C stack can, on
    /// every version. Raises nothing.
    fn release(slf: &Bound<'_, Self>) {
        let freed = refcount(slf) <= 1;
        if let Ok(mut walker) = slf.try_borrow_mut() {
            walker.state.release(slf.py(), freed);
        }
    }

    /// Pickles and copies as `None`, which a context that captures in
    /// Python holds, after a [`Walker::flush`].
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        Self::flush(slf)?;
        let py = slf.py();
        Ok((py.None().into_bound(py).get_type(), PyTuple::empty(py)))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.state.traverse(&visit)
    }

    fn __clear__(&mut self) {
        self.state.clear();
    }
}
