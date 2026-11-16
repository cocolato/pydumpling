//! `ValuesMixin._scalar` and `_fill_copy`: the copy of a str, bytes, int,
//! or float subclass instance, and the type name, attributes, slots, and
//! named fields every subclass copy keeps beside its entries or value, the
//! last two as [`crate::layouts`] lists them.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyTuple};

use crate::keys::pair;
use crate::state::State;
use crate::statics::{type_address, Statics};
use crate::types::{dict_getter, fresh_dict, Base, Layout, Scalar};
use crate::walk::{unless_exception, update};
use crate::walker::Walk;

impl State {
    /// `ValuesMixin._scalar`: a str, bytes, int, or float subclass copy.
    pub(crate) fn scalar<'py>(
        &mut self,
        py: Python<'py>,
        id: &Bound<'py, PyInt>,
        value: &Bound<'py, PyAny>,
        scalar: Scalar,
        depth: i64,
    ) -> Walk<Bound<'py, PyAny>> {
        let s = self.s;
        let limit = self.max_string_bytes;
        let plain = match scalar {
            Scalar::Int => {
                if let Some(limit) = limit {
                    let bits: u64 = s.int_bit_length.bind(py).call1((value,))?.extract()?;
                    if u128::from(bits) > u128::from(limit) * 8 {
                        let omitted = self.omit(py, intern!(py, "max_string_bytes"), value)?;
                        return self.remember(py, id, value, omitted);
                    }
                }
                s.int_int.bind(py).call1((value,))?
            }
            Scalar::Float => s.float_float.bind(py).call1((value,))?,
            Scalar::Str | Scalar::Bytes => {
                // The base slice copies at most one item past the budget.
                let stop = limit.map(|limit| u128::from(limit) + 1);
                let head_slice = s.slice_cls.bind(py).call1((stop,))?;
                let getitem = if scalar == Scalar::Str {
                    &s.str_getitem
                } else {
                    &s.bytes_getitem
                };
                let head = getitem.bind(py).call1((value, head_slice))?;
                if scalar == Scalar::Str {
                    self.text(py, &head)?
                } else {
                    self.bounded_bytes(py, &head)?
                }
            }
        };
        let class = s.scalar_copy(scalar);
        let captured = self.remember(py, id, value, class.make(py, Some(&plain))?)?;
        self.fill_copy(py, value, &captured, class, depth)?;
        Ok(captured)
    }

    /// `ValuesMixin._fill_copy`: a subclass's type name and attributes.
    pub(crate) fn fill_copy<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
        copy: &Bound<'py, PyAny>,
        class: &CopyClass,
        depth: i64,
    ) -> Walk<()> {
        let namespace = &class.namespace(copy)?;
        let cls = value.get_type();
        let own = cls.is(class.cls.bind(py));
        let key = class.key.bind(py).clone();
        let attrs = match self.attributes(py, value)? {
            Some(attrs) if !attrs.is_empty() => attrs,
            _ => PyDict::new(py),
        };
        let type_name = if own {
            // The lookup compares the name with every key of its hash.
            let found = if self.s.str_keys(&attrs) {
                attrs.get_item(&key)?
            } else {
                self.application(py, || attrs.get_item(&key))?
            };
            found.unwrap_or_else(|| py.None().into_bound(py))
        } else {
            self.type_name(py, value)?
        };
        if self.depth_reached(depth) {
            // Only a scalar copy gets here: its attributes and slots are
            // nested data and follow the depth budget.
            let mut nested = false;
            for name in attrs.try_iter()? {
                let name = name?;
                if type_address(&name) != self.s.str_ || name.ne(&key)? {
                    nested = true;
                    break;
                }
            }
            if nested || self.filled_slot(py, value)? {
                self.count(py, intern!(py, "max_depth"))?;
            }
            namespace.set_item(&key, &type_name)?;
            return Ok(());
        }
        if !attrs.is_empty() {
            let private = own.then_some(&key);
            let converted = self.attribute_mapping(py, attrs.as_any(), depth + 1, private)?;
            update(namespace, &converted)?;
        }
        // The namespace holds copies and plain keys until a slot or field
        // name that is not plain goes in ([`State::insert`]).
        let mut mixed = false;
        match &*self.layout(py, value)? {
            Layout::Plain { members, fields } => {
                for (name, getter) in members {
                    let got = getter.bind(py).call1((value, &cls));
                    let Some(item) = unless_exception(py, got)? else {
                        continue; // An unset slot holds nothing.
                    };
                    if !self.slot_fits(py, namespace)? {
                        break;
                    }
                    let converted = self.value(py, &item, depth + 1)?;
                    self.insert(py, namespace, name.bind(py), &converted, &mut mixed)?;
                }
                if !fields.is_empty() && type_address(copy) == self.s.fake_tuple_type {
                    let items = copy.cast::<PyTuple>().map_err(PyErr::from)?;
                    for (field, item) in fields.iter().zip(items.iter()) {
                        self.insert(py, namespace, field.bind(py), &item, &mut mixed)?;
                    }
                }
            }
            Layout::Other { members, fields } => {
                let (members, fields) = (members.bind(py), fields.bind(py));
                self.fill_other(py, value, copy, namespace, members, fields, depth)?;
                // Its names are whatever Python code wrote.
                mixed = true;
            }
        }
        self.insert(py, namespace, &key, &type_name, &mut mixed)
    }

    /// Whether one more slot fits the item and node budgets; counts the one
    /// that stops the copy.
    fn slot_fits(&mut self, py: Python<'_>, namespace: &Bound<'_, PyDict>) -> Walk<bool> {
        if self
            .max_items
            .is_some_and(|limit| namespace.len() as u64 >= limit)
        {
            self.count(py, intern!(py, "max_items"))?;
            return Ok(false);
        }
        if !self.has_nodes(py)? {
            self.count(py, intern!(py, "max_nodes"))?;
            return Ok(false);
        }
        Ok(true)
    }

    /// The slots and fields of a subclass copy, read from a layout
    /// `_layout` did not write: `for name, member in members`,
    /// `member.__get__(value, cls)` and `namespace.update(zip(fields,
    /// copy))`, each running the objects' own code, as in the Python walk.
    #[allow(clippy::too_many_arguments)]
    fn fill_other<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
        copy: &Bound<'py, PyAny>,
        namespace: &Bound<'py, PyDict>,
        members: &Bound<'py, PyAny>,
        fields: &Bound<'py, PyAny>,
        depth: i64,
    ) -> Walk<()> {
        let (s, cls) = (self.s, value.get_type());
        let mut members = self.application(py, || members.try_iter())?;
        while let Some(entry) = self.application(py, || members.next().transpose())? {
            let got = self.application(py, || {
                let (name, member) = pair(s, entry)?;
                let got = member.call_method1(intern!(py, "__get__"), (value, &cls));
                Ok((name, got))
            })?;
            let (name, Some(item)) = (got.0, unless_exception(py, got.1)?) else {
                continue;
            };
            if !self.slot_fits(py, namespace)? {
                break;
            }
            let converted = self.value(py, &item, depth + 1)?;
            self.application(py, || namespace.set_item(&name, converted))?;
        }
        let zip = self.s.zip.bind(py);
        self.application(py, || {
            if fields.is_truthy()? && type_address(copy) == self.s.fake_tuple_type {
                namespace.call_method1(intern!(py, "update"), (zip.call1((fields, copy))?,))?;
            }
            Ok(())
        })
    }
}

/// A snapshot copy class (`FakeDict`, `FakeStr`, ...) as at import: the
/// walker builds its instances with the builtin base's `__new__`, as calling
/// the class does while it holds no hook of its own, and reads their
/// instance dict through the descriptor and their type name under the
/// `_type_key` the class had then, so no hook application code gave the
/// class since runs (the Python walk runs them).
pub(crate) struct CopyClass {
    pub cls: Py<PyAny>,
    new: Py<PyAny>,
    dict: Py<PyAny>,
    key: Py<PyAny>,
}

impl CopyClass {
    pub(crate) fn new(cls: Bound<'_, PyAny>, base: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = cls.py();
        Ok(Self {
            new: base.getattr(intern!(py, "__new__"))?.unbind(),
            dict: dict_getter(&cls)?.unbind(),
            key: cls.getattr(intern!(py, "_type_key"))?.unbind(),
            cls: cls.unbind(),
        })
    }

    /// `cls()`, or `cls(arg)`.
    pub(crate) fn make<'py>(
        &self,
        py: Python<'py>,
        arg: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (new, cls) = (self.new.bind(py), self.cls.bind(py));
        match arg {
            Some(arg) => new.call1((cls, arg)),
            None => new.call1((cls,)),
        }
    }

    /// `copy.__dict__`.
    fn namespace<'py>(&self, copy: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
        fresh_dict(&self.dict, copy)
    }
}

impl Statics {
    /// The copy type for exact instances of `base` and the snapshot subclass
    /// for instances of its subclasses; every frozendict copy is a FakeDict.
    pub(crate) fn copy_types<'py>(
        &self,
        py: Python<'py>,
        base: Base,
    ) -> (Bound<'py, PyAny>, &CopyClass) {
        let snapshot = match base {
            Base::Dict | Base::FrozenDict => &self.fake_dict,
            Base::List => &self.fake_list,
            Base::Tuple => &self.fake_tuple,
            Base::Set => &self.fake_set,
            Base::FrozenSet => &self.fake_frozen_set,
        };
        let plain = match base {
            Base::FrozenDict => snapshot.cls.bind(py).clone(),
            _ => self.base_cls(py, base),
        };
        (plain, snapshot)
    }

    /// The snapshot subclass for instances of a scalar subclass.
    pub(crate) fn scalar_copy(&self, scalar: Scalar) -> &CopyClass {
        match scalar {
            Scalar::Str => &self.fake_str,
            Scalar::Bytes => &self.fake_bytes,
            Scalar::Int => &self.fake_int,
            Scalar::Float => &self.fake_float,
        }
    }

    /// `container()` for `container`, `dict` or `FakeDict`.
    pub(crate) fn empty_mapping<'py>(
        &self,
        py: Python<'py>,
        container: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if container.is(self.fake_dict.cls.bind(py)) {
            self.fake_dict.make(py, None)
        } else {
            container.call0()
        }
    }
}
