//! Python objects the walker compares against or calls.
//!
//! Types are compared by address, as the Python walk compares them with
//! `is`. Unbound methods such as `dict.items` are called on values directly,
//! so a subclass override never runs, exactly as in the Python walk.
//!
//! The table is read from the namespace of the `values` module that builds
//! it, never by importing `pydumpling` by name, so each import of the
//! package, a reloaded one or a vendored copy, gets its own classes, and
//! its own native entries, the table's `value` and `mapping` methods.

use std::sync::{Mutex, OnceLock};

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyTuple};

use crate::address::AddressMap;
use crate::copies::CopyClass;
use crate::types::{class_dicts, dict_getter, Base, Scalar};

/// An object's address: its `id()`, and the identity types are compared by.
pub(crate) fn address(object: &Bound<'_, PyAny>) -> usize {
    object.as_ptr() as usize
}

/// `id(object)`: the key the walk memoizes the object under.
pub(crate) fn id<'py>(object: &Bound<'py, PyAny>) -> Bound<'py, PyInt> {
    match address(object).into_pyobject(object.py()) {
        Ok(id) => id,
        Err(never) => match never {},
    }
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
    pub member_descriptor: usize,
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
    pub fake_dict: CopyClass,
    pub fake_list: CopyClass,
    pub fake_tuple: CopyClass,
    pub fake_set: CopyClass,
    pub fake_frozen_set: CopyClass,
    pub fake_str: CopyClass,
    pub fake_bytes: CopyClass,
    pub fake_int: CopyClass,
    pub fake_float: CopyClass,
    pub fake_class: Py<PyAny>,
    pub fake_class_repr: Py<PyAny>,
    /// `FakeClass` as at import: the classes of its MRO, which assigning
    /// `__bases__` changes, and their dicts but `object`'s, which held only
    /// exact str keys then ([`crate::types::FakeClassNow`]).
    pub fake_class_mro: Vec<Py<PyAny>>,
    pub fake_class_dicts: Vec<Py<PyAny>>,
    pub truncated: Py<PyAny>,
    /// The copy classes' instance-dict getters, from [`dict_getter`].
    pub truncated_dict: Py<PyAny>,
    pub fake_class_dict: Py<PyAny>,
    pub object_new: Py<PyAny>,
    pub dict_items: Py<PyAny>,
    pub dict_keys: Py<PyAny>,
    pub frozendict_items: Option<Py<PyAny>>,
    pub ordered_iter: Py<PyAny>,
    pub set_iter: Py<PyAny>,
    pub frozenset_iter: Py<PyAny>,
    pub int_bit_length: Py<PyAny>,
    /// `operator.iadd`, the in-place add of a tally that is not an exact
    /// int, and `zip`, for a layout `_layout` did not write.
    pub iadd: Py<PyAny>,
    pub zip: Py<PyAny>,
    pub int_int: Py<PyAny>,
    pub float_float: Py<PyAny>,
    pub str_getitem: Py<PyAny>,
    pub bytes_getitem: Py<PyAny>,
    /// Bound `__get__` of `type.__dict__["__dict__"]` and its siblings.
    pub class_dict: Py<PyAny>,
    pub class_mro: Py<PyAny>,
    pub class_module: Py<PyAny>,
    pub class_qualname: Py<PyAny>,
    /// The reason of an immutable container's placeholder: the very object
    /// the Python walk uses, so placeholders pickle identically.
    pub recursive_immutable: Py<PyAny>,
    /// The instance-dict getter of `ValuesMixin`, which reads any
    /// context's instance dict, from [`dict_getter`]: for the one mapping
    /// the walk compares with it ([`crate::mapping`]).
    pub instance_dict: Py<PyAny>,
    /// `gc.get_referents`, which walks an object's references without
    /// making its instance dict: whether that dict exists already.
    pub referents: Py<PyAny>,
    /// `object`.
    pub object: Py<PyAny>,
    /// `DumpOptions`, whether the walker reads its instances' budgets
    /// ([`Statics::reads_options`]), as at import, and the same for other
    /// options types met since, each kept alive with its verdict, a bounded
    /// number at a time.
    pub dump_options: usize,
    pub(crate) dump_options_read: bool,
    pub(crate) options_types: Mutex<AddressMap<(Py<PyAny>, bool)>>,
    /// `collections.Counter`, the only truncation tally type the walker
    /// mirrors.
    pub counter: usize,
    /// `values._IN_PYTHON`, which an entry returns for a call the Python
    /// method runs itself.
    pub in_python: Py<PyAny>,
    /// `ValuesMixin.value`, the Python walk's function, which the walk
    /// calls in place for a node it sends there.
    pub value_function: Py<PyAny>,
    /// The type of a function frame's `f_locals`: `FrameLocalsProxy` on
    /// Python 3.13+, whose `items()` runs no application code. Zero when
    /// frame inspection is unavailable, so proxies use the Python walk.
    pub frame_locals: usize,
    pub true_: Py<PyAny>,
    pub false_: Py<PyAny>,
    /// `type`.
    pub type_: usize,
    /// The composed context class, `SnapshotContext`, once `values._stock`
    /// named it ([`Statics::judged`]).
    pub(crate) stock: OnceLock<Py<PyAny>>,
}

impl Statics {
    pub(crate) fn load(values: &Bound<'_, PyDict>, identity: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = values.py();
        let builtins = py.import("builtins")?;
        let types = py.import("types")?;
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
        let frozendict = builtins.getattr("frozendict").ok();
        let none = py.None().into_bound(py).get_type().into_any();
        let fake_class = fake_type("FakeClass")?;
        let options = global("DumpOptions")?;
        let (class_mro, class_dict) = (getter("_CLASS_MRO")?, getter("_CLASS_DICT")?);
        let mixin = global("ValuesMixin")?;
        // Read from the class dict: the function itself, as the Python
        // walk's own `self.value` calls resolve to it on a native context.
        let value_function = class_dict
            .call1((&mixin, mixin.get_type()))?
            .get_item("value")?;
        let plain_keys = [
            address(&builtin("str")?),
            address(&builtin("int")?),
            address(&builtin("float")?),
            address(&builtin("complex")?),
            address(&builtin("bool")?),
            address(&builtin("bytes")?),
            address(&none),
        ];
        let mut table = Self {
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
            member_descriptor: address(&types.getattr("MemberDescriptorType")?),
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
            fake_dict: CopyClass::new(fake_type("FakeDict")?, &builtin("dict")?)?,
            fake_list: CopyClass::new(fake_type("FakeList")?, &builtin("list")?)?,
            fake_tuple: CopyClass::new(fake_type("FakeTuple")?, &builtin("tuple")?)?,
            fake_set: CopyClass::new(fake_type("FakeSet")?, &builtin("set")?)?,
            fake_frozen_set: CopyClass::new(fake_type("FakeFrozenSet")?, &builtin("frozenset")?)?,
            fake_str: CopyClass::new(fake_type("FakeStr")?, &builtin("str")?)?,
            fake_bytes: CopyClass::new(fake_type("FakeBytes")?, &builtin("bytes")?)?,
            fake_int: CopyClass::new(fake_type("FakeInt")?, &builtin("int")?)?,
            fake_float: CopyClass::new(fake_type("FakeFloat")?, &builtin("float")?)?,
            fake_class_repr: fake_class.getattr("__repr__")?.unbind(),
            fake_class_mro: Vec::new(),
            fake_class_dicts: Vec::new(),
            fake_class_dict: dict_getter(&fake_class)?.unbind(),
            fake_class: fake_class.unbind(),
            truncated_dict: dict_getter(&fake_type("TruncatedValue")?)?.unbind(),
            truncated: fake_type("TruncatedValue")?.unbind(),
            object_new: method("object", "__new__")?.unbind(),
            dict_items: method("dict", "items")?.unbind(),
            dict_keys: method("dict", "keys")?.unbind(),
            ordered_iter: ordered.getattr("__iter__")?.unbind(),
            set_iter: method("set", "__iter__")?.unbind(),
            frozenset_iter: method("frozenset", "__iter__")?.unbind(),
            int_bit_length: method("int", "bit_length")?.unbind(),
            iadd: py.import("operator")?.getattr("iadd")?.unbind(),
            zip: builtin("zip")?.unbind(),
            int_int: method("int", "__int__")?.unbind(),
            float_float: method("float", "__float__")?.unbind(),
            str_getitem: method("str", "__getitem__")?.unbind(),
            bytes_getitem: method("bytes", "__getitem__")?.unbind(),
            class_dict: class_dict.unbind(),
            class_mro: class_mro.unbind(),
            class_module: getter("_CLASS_MODULE")?.unbind(),
            class_qualname: getter("_CLASS_QUALNAME")?.unbind(),
            recursive_immutable: global("_RECURSIVE_IMMUTABLE")?.unbind(),
            instance_dict: dict_getter(&mixin)?.unbind(),
            referents: py.import("gc")?.getattr("get_referents")?.unbind(),
            object: builtin("object")?.unbind(),
            dump_options: address(&options),
            dump_options_read: false,
            options_types: Mutex::new(AddressMap::default()),
            counter: address(&py.import("collections")?.getattr("Counter")?),
            in_python: global("_IN_PYTHON")?.unbind(),
            value_function: value_function.unbind(),
            frame_locals,
            true_: builtin("True")?.unbind(),
            false_: builtin("False")?.unbind(),
            type_: address(&builtin("type")?),
            stock: OnceLock::new(),
        };
        table.dump_options_read = table.options_verdict(&options)?;
        for (base, namespace) in class_dicts(&table, table.fake_class.bind(py))? {
            table.fake_class_mro.push(base.clone().unbind());
            if base.is(table.object.bind(py)) {
                continue;
            }
            for key in namespace.try_iter()? {
                if type_address(&key?) != table.str_ {
                    return Err(PyTypeError::new_err("a FakeClass dict key is not a str"));
                }
            }
            table.fake_class_dicts.push(namespace.unbind());
        }
        Ok(table)
    }

    /// Whether the class of `ctx` was judged to capture natively: the
    /// composed class, or a class `_choose_walk` judged, which holds
    /// `_native_walk_class` naming itself. A class created past
    /// `ValuesMixin.__init_subclass__` (a base ahead of it whose own
    /// `__init_subclass__` does not call `super()`) inherits its flag from
    /// a class that was judged; until `SnapshotContext.__init__` judges it,
    /// an instance that reached it otherwise (unpickled, `object.__new__`,
    /// `__class__` assigned) finds a parent's name there. The name is read
    /// as `type` reads a class attribute, through the MRO and its cache,
    /// when the metaclass is `type`, and from the class's own dict, read
    /// through `type.__dict__`, past any other metaclass's hooks.
    pub(crate) fn judged(&self, ctx: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = ctx.py();
        let cls = ctx.get_type().into_any();
        if self.stock.get().is_some_and(|stock| cls.is(stock.bind(py))) {
            return Ok(true);
        }
        let name = intern!(py, "_native_walk_class");
        let found = if type_address(&cls) == self.type_ {
            cls.getattr_opt(name)?
        } else {
            let namespace = self.class_dict.bind(py).call1((&cls, cls.get_type()))?;
            Some(namespace.call_method1(intern!(py, "get"), (name,))?)
        };
        Ok(found.is_some_and(|found| found.is(&cls)))
    }

    /// `_plain_key`: whether hashing or comparing a key runs no
    /// application code.
    pub(crate) fn plain_key(&self, key: &Bound<'_, PyAny>) -> bool {
        let plain = |item: &Bound<'_, PyAny>| self.plain_keys.contains(&type_address(item));
        match key.cast::<PyTuple>() {
            Ok(items) if type_address(key) == self.tuple => {
                items.iter_borrowed().all(|item| plain(&item))
            }
            _ => plain(key),
        }
    }

    /// Whether every key of `dict` is an exact str, so looking a str up in
    /// it runs no code and compares only strs.
    pub(crate) fn str_keys(&self, dict: &Bound<'_, PyDict>) -> bool {
        dict.iter().all(|(key, _)| type_address(&key) == self.str_)
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
}
