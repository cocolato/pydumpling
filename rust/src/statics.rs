//! Python objects the walker compares against or calls.
//!
//! Types are compared by address, as the Python walk compares them with
//! `is`. Unbound methods such as `dict.items` are called on values directly,
//! so a subclass override never runs, exactly as in the Python walk.
//!
//! The table is read from the namespace of the `values` module that builds
//! it, never by importing `pydumpling` by name, so each import of the
//! package, a reloaded one or a vendored copy, gets its own classes.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};

use crate::types::{Base, Scalar};

/// An object's address: its `id()`, and the identity types are compared by.
pub(crate) fn address(object: &Bound<'_, PyAny>) -> usize {
    object.as_ptr() as usize
}

/// The address of an object's type.
pub(crate) fn type_address(object: &Bound<'_, PyAny>) -> usize {
    object.get_type_ptr() as usize
}

/// Everything the walk looks up by identity, loaded from the same modules
/// the Python walk imports.
pub(crate) struct Statics {
    pub none: usize,
    pub bool_: usize,
    pub int: usize,
    pub float: usize,
    pub complex: usize,
    pub str_: usize,
    pub bytes: usize,
    pub bytearray: usize,
    pub dict: usize,
    pub list: usize,
    pub tuple: usize,
    pub set: usize,
    pub frozenset: usize,
    pub frozendict: Option<usize>,
    pub ordered_dict: usize,
    pub getset_descriptor: usize,
    pub truncated_type: usize,
    pub fake_class_type: usize,
    pub fake_dict_type: usize,
    pub fake_tuple_type: usize,
    /// Key types whose hashing and comparison run only builtin code.
    pub plain_keys: [usize; 7],
    pub dict_cls: Py<PyAny>,
    pub list_cls: Py<PyAny>,
    pub tuple_cls: Py<PyAny>,
    pub set_cls: Py<PyAny>,
    pub frozenset_cls: Py<PyAny>,
    pub frozendict_cls: Option<Py<PyAny>>,
    pub slice_cls: Py<PyAny>,
    pub fake_dict: Py<PyAny>,
    pub fake_list: Py<PyAny>,
    pub fake_tuple: Py<PyAny>,
    pub fake_set: Py<PyAny>,
    pub fake_frozen_set: Py<PyAny>,
    pub fake_str: Py<PyAny>,
    pub fake_bytes: Py<PyAny>,
    pub fake_int: Py<PyAny>,
    pub fake_float: Py<PyAny>,
    pub fake_class: Py<PyAny>,
    pub fake_class_repr: Py<PyAny>,
    pub truncated: Py<PyAny>,
    pub object_new: Py<PyAny>,
    pub dict_items: Py<PyAny>,
    pub dict_keys: Py<PyAny>,
    pub frozendict_items: Option<Py<PyAny>>,
    pub ordered_iter: Py<PyAny>,
    pub set_iter: Py<PyAny>,
    pub frozenset_iter: Py<PyAny>,
    pub int_bit_length: Py<PyAny>,
    pub int_int: Py<PyAny>,
    pub float_float: Py<PyAny>,
    pub str_getitem: Py<PyAny>,
    pub bytes_getitem: Py<PyAny>,
    /// Bound `__get__` of `type.__dict__["__dict__"]` and its siblings.
    pub class_dict: Py<PyAny>,
    pub class_mro: Py<PyAny>,
    pub class_flags: Py<PyAny>,
    pub class_module: Py<PyAny>,
    pub class_qualname: Py<PyAny>,
    /// The reason of an immutable container's placeholder: the very object
    /// the Python walk uses, so placeholders pickle identically.
    pub recursive_immutable: Py<PyAny>,
    /// `sys.getrecursionlimit` and `sys._getframe`.
    pub recursion_limit: Py<PyAny>,
    pub get_frame: Py<PyAny>,
    /// `sys.monitoring._all_events`, where the interpreter has it, else
    /// `sys.getprofile` and `sys.gettrace`, and `sys.monitoring.get_tool`:
    /// whether running Python code would also run a tracing hook.
    pub all_events: Option<Py<PyAny>>,
    pub get_profile: Py<PyAny>,
    pub get_trace: Py<PyAny>,
    pub get_tool: Py<PyAny>,
    /// `object.__getattribute__` and `object.__setattr__`: attribute
    /// access as a stock context's, without its class's hooks.
    pub object_getattribute: Py<PyAny>,
    pub object_setattr: Py<PyAny>,
    /// Whether the collector leaves a dict untracked while it holds nothing
    /// it tracks, as before CPython 3.14: see [`crate::release`].
    pub untracked_dicts: bool,
    /// `values._headroom`, which recurses as deep as it is asked to.
    pub headroom: Py<PyAny>,
    /// `DumpOptions` and `collections.Counter`, the only options and
    /// truncation tally types the walker mirrors.
    pub dump_options: usize,
    pub counter: usize,
    /// `values._NATIVE_WALK`, the context methods the walker stands in
    /// for, then `values._STATE_ACCESS`, `values._NATIVE_STATE` and
    /// `values._COPY_PROTOCOL`, the class attributes the context must hold
    /// as the stock class does, interned.
    pub walk_keys: Vec<Py<PyString>>,
    /// How many of [`Statics::walk_keys`] are `_NATIVE_WALK`, and where
    /// `_COPY_PROTOCOL` starts: the names an instance attribute may shadow.
    pub walk_count: usize,
    pub copy_from: usize,
    /// The names a metaclass may answer the class lookups of `copy.copy`
    /// and `copy.replace` with; `type` holds none of its own.
    pub meta_keys: Vec<Py<PyString>>,
    /// The type of a function frame's `f_locals`: `FrameLocalsProxy` on
    /// Python 3.13+, whose `items()` runs no application code. Zero when
    /// frame inspection is unavailable, so proxies use the Python walk.
    pub frame_locals: usize,
    pub true_: Py<PyAny>,
    pub false_: Py<PyAny>,
}

/// The walker's table for one import of `pydumpling.snapshot.values`.
///
/// `values` builds one when it is imported and hands it to every `Walker`.
/// The table lives as long as the process: a reloaded package builds a new
/// one, and the old one keeps the classes its snapshots were made of.
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
}

impl Statics {
    fn load(values: &Bound<'_, PyDict>, identity: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = values.py();
        let builtins = py.import("builtins")?;
        let types = py.import("types")?;
        let sys = py.import("sys")?;
        let global = |name: &str| -> PyResult<Bound<'_, PyAny>> {
            values.get_item(name)?.ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(format!("values has no {name}"))
            })
        };
        let ordered = global("OrderedDict")?;
        let builtin = |name: &str| builtins.getattr(name);
        let fake_type = |name: &str| global(name);
        let method = |owner: &str, name: &str| builtins.getattr(owner)?.getattr(name);
        let getter = |name: &str| identity.getattr(name)?.getattr("__get__");
        // A denied frame probe disables only the locals-proxy fast path;
        // the Python walk can still read a proxy passed in by its caller.
        let locals_type = global("_frame_locals_type")?.call0()?;
        let frame_locals = if locals_type.is_none() {
            0
        } else {
            address(&locals_type)
        };
        let walk: Vec<String> = global("_NATIVE_WALK")?.extract()?;
        let access: Vec<String> = global("_STATE_ACCESS")?.extract()?;
        let state: Vec<String> = global("_NATIVE_STATE")?.extract()?;
        let copy: Vec<String> = global("_COPY_PROTOCOL")?.extract()?;
        let names: Vec<&String> = walk
            .iter()
            .chain(&access)
            .chain(&state)
            .chain(&copy)
            .collect();
        let copy_from = names.len() - copy.len();
        let walk_keys = names
            .iter()
            .map(|name| PyString::intern(py, name).unbind())
            .collect();
        let type_dict = builtin("type")?.getattr("__dict__")?;
        let frozendict = builtins.getattr("frozendict").ok();
        let none = py.None().into_bound(py).get_type().into_any();
        let fake_class = fake_type("FakeClass")?;
        let plain_keys = [
            address(&builtin("str")?),
            address(&builtin("int")?),
            address(&builtin("float")?),
            address(&builtin("complex")?),
            address(&builtin("bool")?),
            address(&builtin("bytes")?),
            address(&none),
        ];
        Ok(Self {
            none: address(&none),
            bool_: address(&builtin("bool")?),
            int: address(&builtin("int")?),
            float: address(&builtin("float")?),
            complex: address(&builtin("complex")?),
            str_: address(&builtin("str")?),
            bytes: address(&builtin("bytes")?),
            bytearray: address(&builtin("bytearray")?),
            dict: address(&builtin("dict")?),
            list: address(&builtin("list")?),
            tuple: address(&builtin("tuple")?),
            set: address(&builtin("set")?),
            frozenset: address(&builtin("frozenset")?),
            frozendict: frozendict.as_ref().map(address),
            ordered_dict: address(&ordered),
            getset_descriptor: address(&types.getattr("GetSetDescriptorType")?),
            truncated_type: address(&fake_type("TruncatedValue")?),
            fake_class_type: address(&fake_class),
            fake_dict_type: address(&fake_type("FakeDict")?),
            fake_tuple_type: address(&fake_type("FakeTuple")?),
            plain_keys,
            dict_cls: builtin("dict")?.unbind(),
            list_cls: builtin("list")?.unbind(),
            tuple_cls: builtin("tuple")?.unbind(),
            set_cls: builtin("set")?.unbind(),
            frozenset_cls: builtin("frozenset")?.unbind(),
            frozendict_items: match &frozendict {
                Some(cls) => Some(cls.getattr("items")?.unbind()),
                None => None,
            },
            frozendict_cls: frozendict.map(Bound::unbind),
            slice_cls: builtin("slice")?.unbind(),
            fake_dict: fake_type("FakeDict")?.unbind(),
            fake_list: fake_type("FakeList")?.unbind(),
            fake_tuple: fake_type("FakeTuple")?.unbind(),
            fake_set: fake_type("FakeSet")?.unbind(),
            fake_frozen_set: fake_type("FakeFrozenSet")?.unbind(),
            fake_str: fake_type("FakeStr")?.unbind(),
            fake_bytes: fake_type("FakeBytes")?.unbind(),
            fake_int: fake_type("FakeInt")?.unbind(),
            fake_float: fake_type("FakeFloat")?.unbind(),
            fake_class_repr: fake_class.getattr("__repr__")?.unbind(),
            fake_class: fake_class.unbind(),
            truncated: fake_type("TruncatedValue")?.unbind(),
            object_new: method("object", "__new__")?.unbind(),
            dict_items: method("dict", "items")?.unbind(),
            dict_keys: method("dict", "keys")?.unbind(),
            ordered_iter: ordered.getattr("__iter__")?.unbind(),
            set_iter: method("set", "__iter__")?.unbind(),
            frozenset_iter: method("frozenset", "__iter__")?.unbind(),
            int_bit_length: method("int", "bit_length")?.unbind(),
            int_int: method("int", "__int__")?.unbind(),
            float_float: method("float", "__float__")?.unbind(),
            str_getitem: method("str", "__getitem__")?.unbind(),
            bytes_getitem: method("bytes", "__getitem__")?.unbind(),
            class_dict: getter("_CLASS_DICT")?.unbind(),
            class_mro: getter("_CLASS_MRO")?.unbind(),
            class_flags: type_dict
                .get_item("__flags__")?
                .getattr("__get__")?
                .unbind(),
            class_module: getter("_CLASS_MODULE")?.unbind(),
            class_qualname: getter("_CLASS_QUALNAME")?.unbind(),
            recursive_immutable: global("_RECURSIVE_IMMUTABLE")?.unbind(),
            recursion_limit: sys.getattr("getrecursionlimit")?.unbind(),
            get_frame: sys.getattr("_getframe")?.unbind(),
            get_profile: sys.getattr("getprofile")?.unbind(),
            get_trace: sys.getattr("gettrace")?.unbind(),
            all_events: sys
                .getattr("monitoring")?
                .getattr("_all_events")
                .ok()
                .map(Bound::unbind),
            get_tool: sys.getattr("monitoring")?.getattr("get_tool")?.unbind(),
            object_getattribute: method("object", "__getattribute__")?.unbind(),
            object_setattr: method("object", "__setattr__")?.unbind(),
            untracked_dicts: !py
                .import("gc")?
                .getattr("is_tracked")?
                .call1((PyDict::new(py),))?
                .is_truthy()?,
            headroom: global("_headroom")?.unbind(),
            dump_options: address(&global("DumpOptions")?),
            counter: address(&py.import("collections")?.getattr("Counter")?),
            walk_keys,
            walk_count: walk.len(),
            copy_from,
            meta_keys: ["__copy__", "__replace__", "__getattr__", "__getattribute__"]
                .map(|name| PyString::intern(py, name).unbind())
                .into(),
            frame_locals,
            true_: builtin("True")?.unbind(),
            false_: builtin("False")?.unbind(),
        })
    }

    /// `_plain_key`: whether hashing or comparing a key runs no
    /// application code.
    pub(crate) fn plain_key(&self, key: &Bound<'_, PyAny>) -> bool {
        let plain = |item: &Bound<'_, PyAny>| self.plain_keys.contains(&type_address(item));
        match key.cast::<PyTuple>() {
            Ok(items) if type_address(key) == self.tuple => items.iter().all(|item| plain(&item)),
            _ => plain(key),
        }
    }

    /// Whether every key of `dict` is plain, so looking a str up in it
    /// compares with builtin keys only.
    pub(crate) fn plain_dict(&self, dict: &Bound<'_, PyDict>) -> bool {
        dict.iter().all(|(key, _)| self.plain_key(&key))
    }

    /// The scalar base a type in an MRO stands for, if any.
    pub(crate) fn scalar_of(&self, address: usize) -> Option<Scalar> {
        match address {
            a if a == self.str_ => Some(Scalar::Str),
            a if a == self.bytes => Some(Scalar::Bytes),
            a if a == self.int => Some(Scalar::Int),
            a if a == self.float => Some(Scalar::Float),
            _ => None,
        }
    }

    /// The container base a type in an MRO stands for, if any.
    pub(crate) fn base_of(&self, address: usize) -> Option<Base> {
        match address {
            a if a == self.dict => Some(Base::Dict),
            a if a == self.list => Some(Base::List),
            a if a == self.tuple => Some(Base::Tuple),
            a if a == self.set => Some(Base::Set),
            a if a == self.frozenset => Some(Base::FrozenSet),
            a if Some(a) == self.frozendict => Some(Base::FrozenDict),
            _ => None,
        }
    }

    /// The builtin base type itself.
    pub(crate) fn base_cls<'py>(&self, py: Python<'py>, base: Base) -> Bound<'py, PyAny> {
        let cls = match base {
            Base::Dict => &self.dict_cls,
            Base::List => &self.list_cls,
            Base::Tuple => &self.tuple_cls,
            Base::Set => &self.set_cls,
            Base::FrozenSet => &self.frozenset_cls,
            Base::FrozenDict => self.frozendict_cls.as_ref().unwrap_or(&self.dict_cls),
        };
        cls.bind(py).clone()
    }

    /// The copy type for exact instances of `base` and the snapshot subclass
    /// for instances of its subclasses; every frozendict copy is a FakeDict.
    pub(crate) fn copy_types<'py>(
        &self,
        py: Python<'py>,
        base: Base,
    ) -> (Bound<'py, PyAny>, Bound<'py, PyAny>) {
        let snapshot = match base {
            Base::Dict | Base::FrozenDict => &self.fake_dict,
            Base::List => &self.fake_list,
            Base::Tuple => &self.fake_tuple,
            Base::Set => &self.fake_set,
            Base::FrozenSet => &self.fake_frozen_set,
        };
        let snapshot = snapshot.bind(py).clone();
        let plain = match base {
            Base::FrozenDict => snapshot.clone(),
            _ => self.base_cls(py, base),
        };
        (plain, snapshot)
    }

    /// The snapshot subclass for instances of a scalar subclass.
    pub(crate) fn scalar_copy<'py>(&self, py: Python<'py>, scalar: Scalar) -> Bound<'py, PyAny> {
        let copy = match scalar {
            Scalar::Str => &self.fake_str,
            Scalar::Bytes => &self.fake_bytes,
            Scalar::Int => &self.fake_int,
            Scalar::Float => &self.fake_float,
        };
        copy.bind(py).clone()
    }
}
