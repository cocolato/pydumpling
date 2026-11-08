//! `ValuesMixin._scalar` and `_fill_copy`: the copy of a str, bytes, int,
//! or float subclass instance, and the type name, attributes, slots, and
//! named fields every subclass copy keeps beside its entries or value.

use std::sync::Arc;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::statics::type_address;
use crate::types::{Layout, Scalar};
use crate::walk::{unless_exception, update};
use crate::walker::{Halt, State, Walk};

impl State {
    /// `ValuesMixin._layout`, called once per type and cached.
    pub(crate) fn layout(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> Walk<Arc<Layout>> {
        let info = self.info(py, &value.get_type())?;
        if info.hostile {
            // `_layout` puts each class dict key in a set.
            return Err(Halt::Bail);
        }
        if let Some(layout) = info.layout.clone() {
            return Ok(layout);
        }
        self.before_python(py)?;
        let pair = self
            .ctx(py)?
            .call_method1(intern!(py, "_layout"), (value,))?;
        let layout = Arc::new(Layout::from_python(py, &pair)?);
        self.info(py, &value.get_type())?.layout = Some(layout.clone());
        Ok(layout)
    }

    /// `ValuesMixin._scalar`: a str, bytes, int, or float subclass copy.
    pub(crate) fn scalar<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
        scalar: Scalar,
        depth: i64,
    ) -> Walk<Bound<'py, PyAny>> {
        self.enter(py)?;
        let captured = self.scalar_body(py, value, scalar, depth);
        self.leave();
        captured
    }

    fn scalar_body<'py>(
        &mut self,
        py: Python<'py>,
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
                        return Ok(self.remember(value, omitted));
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
        let copy = s.scalar_copy(py, scalar).call1((plain,))?;
        let captured = self.remember(value, copy);
        self.fill_copy(py, value, &captured, depth)?;
        Ok(captured)
    }

    /// `ValuesMixin._fill_copy`: a subclass's type name and attributes.
    pub(crate) fn fill_copy<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
        copy: &Bound<'py, PyAny>,
        depth: i64,
    ) -> Walk<()> {
        self.enter(py)?;
        let filled = self.fill_copy_body(py, value, copy, depth);
        self.leave();
        filled
    }

    fn fill_copy_body<'py>(
        &mut self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
        copy: &Bound<'py, PyAny>,
        depth: i64,
    ) -> Walk<()> {
        let namespace = copy.getattr(intern!(py, "__dict__"))?;
        let namespace = namespace.cast::<PyDict>().map_err(PyErr::from)?;
        let cls = value.get_type();
        let copy_type = copy.get_type();
        let own = cls.is(&copy_type);
        let key = copy_type.getattr(intern!(py, "_type_key"))?;
        let attrs = match self.attributes(py, value)? {
            Some(attrs) if !attrs.is_empty() => attrs,
            _ => PyDict::new(py),
        };
        let type_name = if own {
            // The lookup compares the name with every key of its hash.
            if !self.s.plain_dict(&attrs) {
                return Err(Halt::Bail);
            }
            attrs
                .get_item(&key)?
                .unwrap_or_else(|| py.None().into_bound(py))
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
        let layout = self.layout(py, value)?;
        for (name, getter) in &layout.members {
            let Some(item) = unless_exception(py, getter.bind(py).call1((value, &cls)))? else {
                continue; // An unset slot holds nothing.
            };
            if self
                .max_items
                .is_some_and(|limit| namespace.len() as u64 >= limit)
            {
                self.count(py, intern!(py, "max_items"))?;
                break;
            }
            if !self.has_nodes() {
                self.count(py, intern!(py, "max_nodes"))?;
                break;
            }
            let converted = self.value(py, &item, depth + 1)?;
            namespace.set_item(name.bind(py), converted)?;
        }
        if !layout.fields.is_empty() && type_address(copy) == self.s.fake_tuple_type {
            let items = copy.cast::<PyTuple>().map_err(PyErr::from)?;
            for (field, item) in layout.fields.iter().zip(items.iter()) {
                namespace.set_item(field.bind(py), item)?;
            }
        }
        namespace.set_item(&key, &type_name)?;
        Ok(())
    }

    /// `ValuesMixin._filled_slot`: whether any slot holds a value.
    fn filled_slot(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> Walk<bool> {
        let layout = self.layout(py, value)?;
        let cls = value.get_type();
        for (_, getter) in &layout.members {
            if unless_exception(py, getter.bind(py).call1((value, &cls)))?.is_some() {
                return Ok(true);
            }
        }
        Ok(false)
    }
}
