//! The context state a native call reads, with ordinary attribute access;
//! inserts into a dict that may hold application keys; and whether the
//! walker reads an options object's budgets.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PySet, PyString};

use crate::address::AddressMap;
use crate::statics::{address, type_address, Statics};
use crate::text::budget;
use crate::types::{class_dicts, FakeClassNow, TypeInfo};
use crate::walker::Walk;

/// The `SnapshotContext` state of one native call. `objects`, `_nodes`,
/// `options` and its budgets are read when the call starts; the rest the
/// first time the call needs each, which is where the Python walk first
/// reads it ([`lazy`]).
pub(crate) struct State {
    pub s: &'static Statics,
    pub(crate) ctx: Py<PyAny>,
    pub(crate) objects: Py<PyDict>,
    pub max_items: Option<u64>,
    pub max_nodes: Option<u64>,
    pub max_depth: Option<u64>,
    pub max_string_bytes: Option<u64>,
    pub nodes: u64,
    /// `_nodes` as the context holds it, if an exact int the walk reads:
    /// the walk writes its count there when it differs.
    pub(crate) written: Option<u64>,
    /// Whether Python code the call ran deleted `_nodes`: the walk reads it
    /// again where the Python walk next reads it, which raises there
    /// ([`State::has_nodes`], [`State::add_node`]), and writes it nowhere.
    pub(crate) missing: bool,
    sources: Option<Py<PyAny>>,
    truncation: Option<Py<PyAny>>,
    node_marker: Option<Py<PyAny>>,
    namespaces: Option<Py<PyAny>>,
    layouts: Option<Py<PyAny>>,
    code_text_limit: Option<Py<PyAny>>,
    /// Whether `_unlimited` is `False`.
    limited: Option<bool>,
    /// Whether the call has done anything a rerun in Python would repeat.
    pub(crate) started: bool,
    /// Facts about the types met in the current call, dropped when
    /// application code runs ([`State::forget_types`]).
    pub types: AddressMap<TypeInfo>,
    /// `FakeClass` as it is now, once the call has asked.
    fake_class: Option<FakeClassNow>,
    /// Whether every key of `_truncation` is an exact str, so looking a
    /// tally up runs no key's `__eq__`.
    truncation_plain: Option<bool>,
    /// The slice `[:limit]` takes, and its limit ([`State::head`]).
    pub(crate) head: Option<(u64, Py<PyAny>)>,
    /// Stack address where the current call entered the walker, and how
    /// much native stack the call may use below it.
    pub(crate) origin: usize,
    pub(crate) stack_budget: usize,
}

/// `ctx.<name>`, read into `slot` the first time a call needs it. A
/// missing attribute raises there, as at the Python walk's read.
fn lazy<'py>(
    slot: &mut Option<Py<PyAny>>,
    ctx: &Py<PyAny>,
    name: &Bound<'py, PyString>,
) -> PyResult<Bound<'py, PyAny>> {
    match slot {
        Some(held) => Ok(held.bind(name.py()).clone()),
        None => read(slot, ctx, name),
    }
}

/// [`lazy`]'s first read, out of line, as the frame of each native level
/// would otherwise hold its locals.
#[cold]
#[inline(never)]
fn read<'py>(
    slot: &mut Option<Py<PyAny>>,
    ctx: &Py<PyAny>,
    name: &Bound<'py, PyString>,
) -> PyResult<Bound<'py, PyAny>> {
    let held = ctx.bind(name.py()).getattr(name)?;
    *slot = Some(held.clone().unbind());
    Ok(held)
}

impl State {
    /// Reads the state every call needs, as stock attribute access reads
    /// it: `objects`, `_nodes` and the four budgets of `options`. `None` if
    /// only the Python walk reproduces it: an attribute missing, `objects`
    /// not an exact dict, `_nodes` not an exact int from 0 to 2**64 - 1, or
    /// options the walker does not read ([`Statics::reads_options`]).
    pub(crate) fn load<'py>(
        py: Python<'py>,
        s: &'static Statics,
        ctx: &Bound<'py, PyAny>,
        stack_budget: usize,
    ) -> PyResult<Option<Self>> {
        let get = |name: &Bound<'py, PyString>| ctx.getattr_opt(name);
        let (Some(objects), Some(count), Some(options)) = (
            get(intern!(py, "objects"))?,
            get(intern!(py, "_nodes"))?,
            get(intern!(py, "options"))?,
        ) else {
            return Ok(None);
        };
        let Ok(objects) = objects.cast_into_exact::<PyDict>() else {
            return Ok(None);
        };
        if type_address(&count) != s.int {
            return Ok(None);
        }
        let Ok(nodes) = count.extract::<u64>() else {
            return Ok(None);
        };
        if !s.reads_options(&options)? {
            return Ok(None);
        }
        let field = |name: &Bound<'py, PyString>| options.getattr_opt(name);
        let (Some(items), Some(limit), Some(depth), Some(string)) = (
            field(intern!(py, "max_items"))?,
            field(intern!(py, "max_nodes"))?,
            field(intern!(py, "max_depth"))?,
            field(intern!(py, "max_string_bytes"))?,
        ) else {
            return Ok(None);
        };
        let (Some(max_items), Some(max_nodes), Some(max_depth), Some(max_string_bytes)) = (
            budget(s, &items),
            budget(s, &limit),
            budget(s, &depth),
            budget(s, &string),
        ) else {
            return Ok(None);
        };
        Ok(Some(Self {
            s,
            ctx: ctx.clone().unbind(),
            objects: objects.unbind(),
            max_items,
            max_nodes,
            max_depth,
            max_string_bytes,
            nodes,
            written: Some(nodes),
            missing: false,
            sources: None,
            truncation: None,
            node_marker: None,
            namespaces: None,
            layouts: None,
            code_text_limit: None,
            limited: None,
            started: false,
            types: AddressMap::default(),
            fake_class: None,
            truncation_plain: None,
            head: None,
            origin: 0,
            stack_budget,
        }))
    }

    /// The snapshot context of the call.
    pub(crate) fn ctx<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.ctx.bind(py)
    }

    /// `ctx.objects`, as the call read it.
    pub(crate) fn objects<'py>(&self, py: Python<'py>) -> &Bound<'py, PyDict> {
        self.objects.bind(py)
    }

    /// Drops what the call learnt about classes, after application code
    /// that may have changed them ran: the next value of each type reads
    /// its facts anew.
    pub(crate) fn forget_types(&mut self) {
        self.types.clear();
        self.fake_class = None;
        self.truncation_plain = None;
    }

    /// `target[key] = value` into a dict that holds application keys once
    /// any key it was given is not of a plain kind (`mixed` says so): from
    /// then on, an insert may compare with such a key, whose `__eq__` runs,
    /// so it runs as application code. Only plain keys before it are
    /// inserted natively.
    pub(crate) fn insert<'py>(
        &mut self,
        py: Python<'py>,
        target: &Bound<'py, PyDict>,
        key: &Bound<'py, PyAny>,
        value: &Bound<'py, PyAny>,
        mixed: &mut bool,
    ) -> Walk<()> {
        if !*mixed && self.s.plain_key(key) {
            return Ok(target.set_item(key, value)?);
        }
        *mixed = true;
        self.application(py, || target.set_item(key, value))
    }

    /// `FakeClass` as it is now ([`FakeClassNow`]).
    pub(crate) fn fake_class(&mut self, py: Python<'_>) -> PyResult<FakeClassNow> {
        if let Some(now) = self.fake_class {
            return Ok(now);
        }
        let now = FakeClassNow::read(py, self.s)?;
        self.fake_class = Some(now);
        Ok(now)
    }

    /// `ctx._sources`, read at the call's first `_remember`.
    pub(crate) fn sources<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        lazy(&mut self.sources, &self.ctx, intern!(py, "_sources"))
    }

    /// `ctx._truncation`, whether it is an exact `Counter`, and whether
    /// that one's keys are all exact strs, so its tallies are read in the
    /// dict running no code.
    pub(crate) fn truncation<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, bool, bool)> {
        let truncation = lazy(&mut self.truncation, &self.ctx, intern!(py, "_truncation"))?;
        if type_address(&truncation) != self.s.counter {
            return Ok((truncation, false, false));
        }
        let plain = match self.truncation_plain {
            Some(plain) => plain,
            None => {
                let plain = self.s.str_keys(truncation.cast::<PyDict>()?);
                self.truncation_plain = Some(plain);
                plain
            }
        };
        Ok((truncation, true, plain))
    }

    /// `ctx._node_marker`.
    pub(crate) fn node_marker<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        lazy(
            &mut self.node_marker,
            &self.ctx,
            intern!(py, "_node_marker"),
        )
    }

    /// `ctx._layouts`.
    pub(crate) fn layouts<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        lazy(&mut self.layouts, &self.ctx, intern!(py, "_layouts"))
    }

    /// `ctx._code_text_limit`.
    pub(crate) fn code_text_limit<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let name = intern!(py, "_code_text_limit");
        lazy(&mut self.code_text_limit, &self.ctx, name)
    }

    /// Whether `ctx._unlimited` is `False`, as on every context the walker
    /// captures; an unlimited one probes each value with the serializer.
    pub(crate) fn limited(&mut self, py: Python<'_>) -> PyResult<bool> {
        match self.limited {
            Some(limited) => Ok(limited),
            None => self.read_limited(py),
        }
    }

    /// [`State::limited`]'s first read, out of line as [`read`].
    #[cold]
    #[inline(never)]
    fn read_limited(&mut self, py: Python<'_>) -> PyResult<bool> {
        let unlimited = self.ctx(py).getattr(intern!(py, "_unlimited"))?;
        let limited = unlimited.is(self.s.false_.bind(py));
        self.limited = Some(limited);
        Ok(limited)
    }

    /// `id in ctx._namespaces`: whether the dict of that id holds names.
    pub(crate) fn named(&mut self, py: Python<'_>, id: &Bound<'_, PyInt>) -> Walk<bool> {
        let namespaces = lazy(&mut self.namespaces, &self.ctx, intern!(py, "_namespaces"))?;
        if type_address(&namespaces) == self.s.set {
            return Ok(namespaces.contains(id)?);
        }
        // Its own `__contains__` may run.
        self.application(py, || namespaces.contains(id))
    }

    /// `ctx._namespaces.add(id)`.
    pub(crate) fn add_named(&mut self, py: Python<'_>, id: &Bound<'_, PyInt>) -> Walk<()> {
        let namespaces = lazy(&mut self.namespaces, &self.ctx, intern!(py, "_namespaces"))?;
        if let Ok(set) = namespaces.cast_exact::<PySet>() {
            return Ok(set.add(id)?);
        }
        let add = intern!(py, "add");
        self.application(py, || namespaces.call_method1(add, (id,)).map(drop))
    }
}

/// How many options types other than `DumpOptions` the table keeps a
/// verdict for, and with it the type alive; past that it forgets them all.
const OPTIONS_TYPES_KEPT: usize = 64;

impl Statics {
    /// Whether the walker reads the budgets of `options` once per call, as
    /// `options.max_items` and the rest read them. The verdict is kept per
    /// options type, for up to [`OPTIONS_TYPES_KEPT`] types at a time: a
    /// class changed later is not seen while its verdict is kept.
    pub(crate) fn reads_options(&self, options: &Bound<'_, PyAny>) -> PyResult<bool> {
        let ty = type_address(options);
        if ty == self.dump_options {
            return Ok(self.dump_options_read);
        }
        let known = |types: &AddressMap<(Py<PyAny>, bool)>| types.get(&ty).map(|kept| kept.1);
        let lock = || {
            self.options_types
                .lock()
                .unwrap_or_else(|err| err.into_inner())
        };
        if let Some(verdict) = known(&lock()) {
            return Ok(verdict);
        }
        // Computed unlocked: reading classes calls into Python.
        let cls = options.get_type().into_any();
        let verdict = self.options_verdict(&cls)?;
        let mut types = lock();
        if types.len() >= OPTIONS_TYPES_KEPT {
            // Each holds its type alive: a process that makes options
            // classes as it goes would keep every one of them.
            types.clear();
        }
        types.insert(ty, (cls.unbind(), verdict));
        Ok(verdict)
    }

    /// Whether an instance of `cls`, a `DumpOptions` subclass or
    /// `DumpOptions`, reads its budgets as `DumpOptions` does, with no code
    /// but the attribute lookup: along its MRO, read through `type.__dict__`
    /// without any metaclass hook, every class dict has only exact str keys,
    /// none holds `__getattribute__` or `__getattr__` (but `object`), and
    /// none in front of `DumpOptions` holds a budget's name.
    pub(crate) fn options_verdict(&self, cls: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = cls.py();
        let names = [
            intern!(py, "max_items"),
            intern!(py, "max_nodes"),
            intern!(py, "max_depth"),
            intern!(py, "max_string_bytes"),
        ];
        let hooks = [intern!(py, "__getattribute__"), intern!(py, "__getattr__")];
        let mut in_front = true;
        for (base, namespace) in class_dicts(self, cls)? {
            if base.is(self.object.bind(py)) {
                continue;
            }
            // Iterating a class dict yields its keys and runs nothing else.
            for key in namespace.try_iter()? {
                if type_address(&key?) != self.str_ {
                    return Ok(false);
                }
            }
            for hook in hooks {
                if namespace.contains(hook)? {
                    return Ok(false);
                }
            }
            if address(&base) == self.dump_options {
                in_front = false;
            } else if in_front {
                for name in names {
                    if namespace.contains(name)? {
                        return Ok(false);
                    }
                }
            }
        }
        Ok(!in_front)
    }
}
