//! `ValuesMixin._layout`: the member descriptors and named tuple fields
//! of a subclass, which `ctx._layouts` caches per class, and
//! `_filled_slot`, which reads them.

use std::sync::Arc;

use pyo3::exceptions::PyException;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple, PyType};

use crate::keys::pair;
use crate::state::State;
use crate::statics::{id, type_address, Statics};
use crate::types::{class_dicts, Layout};
use crate::walk::unless_exception;
use crate::walker::Walk;

impl State {
    /// `ValuesMixin._layout`: what `ctx._layouts` holds for `type(value)`,
    /// computed and written there as the helper does when it holds none.
    /// `_layouts` other than an exact dict runs its own `get` and
    /// `__setitem__`, as in the Python walk, for every value. An exact
    /// dict is read and written natively, and a layout `_layout` wrote
    /// there is kept per type until the walker next syncs: a key planted
    /// there that hashes like a class id runs its `__eq__` unsynced, at the
    /// walker's lookups only (`rust/DESIGN.md`, "The count is stale between
    /// synced points").
    pub(crate) fn layout(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> Walk<Arc<Layout>> {
        let cls = value.get_type();
        let layouts = self.layouts(py)?;
        let plain = layouts.cast_exact::<PyDict>().ok();
        if plain.is_some() {
            if let Some(layout) = self.info(py, &cls)?.layout.clone() {
                return Ok(layout);
            }
        }
        let key = id(cls.as_any());
        let kept = match plain {
            Some(layouts) => layouts.get_item(&key)?,
            None => {
                let get = intern!(py, "get");
                Some(self.application(py, || layouts.call_method1(get, (&key,)))?)
            }
        };
        let plain = plain.is_some();
        let kept = match kept {
            Some(kept) if !kept.is_none() => kept,
            _ => self.write_layout(py, &cls, &layouts, plain)?,
        };
        // `layout[1], layout[2]`, which on anything but a tuple `_layout`
        // wrote may run code.
        let (members, fields) = match kept.cast_exact::<PyTuple>() {
            Ok(kept) if kept.len() >= 3 => (kept.get_item(1)?, kept.get_item(2)?),
            _ => self.application(py, || Ok((kept.get_item(1)?, kept.get_item(2)?)))?,
        };
        let layout = Arc::new(Layout::read(self.s, members, fields)?);
        if plain && matches!(*layout, Layout::Plain { .. }) {
            self.info(py, &cls)?.layout = Some(layout.clone());
        }
        Ok(layout)
    }

    /// `_layout`'s computation for `cls`, and `self._layouts[id(cls)] =
    /// (cls, members, fields)`: the first definition of each name along the
    /// MRO, tracked in a set, as attribute lookup takes it. A class dict key
    /// that is not an exact str runs its own `__hash__` and `__eq__` there,
    /// and in the comparison with `"_fields"`, as often as in the helper.
    fn write_layout<'py>(
        &mut self,
        py: Python<'py>,
        cls: &Bound<'py, PyType>,
        layouts: &Bound<'py, PyAny>,
        plain: bool,
    ) -> Walk<Bound<'py, PyAny>> {
        let s = self.s;
        let compute = || -> PyResult<Bound<'py, PyAny>> {
            let members = PyList::empty(py);
            let mut fields = PyTuple::empty(py).into_any();
            let seen = PySet::empty(py)?;
            for (_, namespace) in class_dicts(s, cls.as_any())? {
                for item in namespace.call_method0(intern!(py, "items"))?.try_iter()? {
                    let (name, member) = pair(s, item?)?;
                    if seen.contains(&name)? {
                        continue;
                    }
                    seen.add(&name)?;
                    if type_address(&member) == s.member_descriptor {
                        members.append((name, member))?;
                    } else if name.eq(intern!(py, "_fields"))? && type_address(&member) == s.tuple {
                        let mut names = member.try_iter()?;
                        if names
                            .all(|field| field.is_ok_and(|field| type_address(&field) == s.str_))
                        {
                            fields = member;
                        }
                    }
                }
            }
            Ok(PyTuple::new(py, [cls.as_any(), members.as_any(), &fields])?.into_any())
        };
        let layout = if self.info(py, cls)?.hostile {
            self.application(py, compute)?
        } else {
            compute()?
        };
        let key = id(cls.as_any());
        if plain {
            layouts.set_item(key, &layout)?;
        } else {
            self.application(py, || layouts.set_item(key, &layout))?;
        }
        Ok(layout)
    }

    /// `ValuesMixin._filled_slot`: whether any slot holds a value.
    pub(crate) fn filled_slot(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> Walk<bool> {
        let layout = self.layout(py, value)?;
        let cls = value.get_type();
        match &*layout {
            Layout::Plain { members, .. } => {
                for (_, getter) in members {
                    if unless_exception(py, getter.bind(py).call1((value, &cls)))?.is_some() {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Layout::Other { members, .. } => {
                let (s, members) = (self.s, members.bind(py));
                let found = self.application(py, || {
                    for entry in members.try_iter()? {
                        let (_, member) = pair(s, entry?)?;
                        match member.call_method1(intern!(py, "__get__"), (value, &cls)) {
                            Ok(_) => return Ok(Ok(true)),
                            Err(err) if err.is_instance_of::<PyException>(py) => continue,
                            Err(err) => return Ok(Err(err)),
                        }
                    }
                    Ok(Ok(false))
                })?;
                Ok(found?)
            }
        }
    }
}

impl Layout {
    /// The layout `_layouts` holds as `members` and `fields`: [`Layout::Plain`]
    /// if they are what `_layout` writes, an exact list of exact pairs of a
    /// name and an exact member descriptor, and an exact tuple of exact strs.
    pub(crate) fn read(
        s: &Statics,
        members: Bound<'_, PyAny>,
        fields: Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let plain = |members: &Bound<'_, PyAny>, fields: &Bound<'_, PyAny>| -> PyResult<bool> {
            if type_address(members) != s.list || type_address(fields) != s.tuple {
                return Ok(false);
            }
            // Iterating an exact list or tuple runs no code.
            for member in members.try_iter()? {
                let member = member?;
                match member.cast_exact::<PyTuple>() {
                    Ok(pair) if pair.len() == 2 => {
                        if type_address(&pair.get_item(1)?) != s.member_descriptor {
                            return Ok(false);
                        }
                    }
                    _ => return Ok(false),
                }
            }
            for field in fields.try_iter()? {
                if type_address(&field?) != s.str_ {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        if !plain(&members, &fields)? {
            return Ok(Layout::Other {
                members: members.unbind(),
                fields: fields.unbind(),
            });
        }
        let py = members.py();
        let mut kept = Vec::new();
        for member in members.try_iter()? {
            let member = member?;
            let getter = member.get_item(1)?.getattr(intern!(py, "__get__"))?;
            kept.push((member.get_item(0)?.unbind(), getter.unbind()));
        }
        let fields = fields
            .try_iter()?
            .map(|field| field.map(Bound::unbind))
            .collect::<PyResult<_>>()?;
        Ok(Layout::Plain {
            members: kept,
            fields,
        })
    }
}
