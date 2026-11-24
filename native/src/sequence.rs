use pyo3::prelude::*;
use pyo3::types::{PyFrozenSet, PyIterator, PyList, PySet, PyTuple};
use pyo3::{intern, PyTypeInfo};

use crate::{Child, Context, Obj, Step};

#[derive(Clone, Copy)]
pub(crate) enum SequenceKind {
    List,
    Tuple,
    Set,
    FrozenSet,
}

impl SequenceKind {
    pub(crate) fn of(value: &Obj<'_>) -> Option<Self> {
        if value.is_exact_instance_of::<PyList>() {
            Some(Self::List)
        } else if value.is_exact_instance_of::<PyTuple>() {
            Some(Self::Tuple)
        } else if value.is_exact_instance_of::<PySet>() {
            Some(Self::Set)
        } else if value.is_exact_instance_of::<PyFrozenSet>() {
            Some(Self::FrozenSet)
        } else {
            None
        }
    }
}

pub(crate) struct Sequence<'py> {
    source: Obj<'py>,
    kind: SequenceKind,
    items: Bound<'py, PyIterator>,
    copy: Obj<'py>,
    depth: i64,
    count: u64,
}

impl<'py> Sequence<'py> {
    pub(crate) fn new(
        ctx: &Context<'py>,
        source: Obj<'py>,
        depth: i64,
        kind: SequenceKind,
    ) -> PyResult<Self> {
        let py = ctx.py;
        let copy = match kind {
            SequenceKind::List | SequenceKind::Tuple => PyList::empty(py).into_any(),
            SequenceKind::Set | SequenceKind::FrozenSet => PySet::empty(py)?.into_any(),
        };
        match kind {
            SequenceKind::Tuple | SequenceKind::FrozenSet => {
                let name = ctx
                    .ctx
                    .call_method1(intern!(py, "_type_name"), (&source,))?;
                let placeholder = ctx.truncated.call1((&ctx.recursive_reason, name))?;
                ctx.remember(&source, &placeholder)?;
            }
            _ => ctx.remember(&source, &copy)?,
        }
        let items = source.try_iter()?;
        Ok(Self {
            source,
            kind,
            items,
            copy,
            depth,
            count: 0,
        })
    }

    pub(crate) fn resume(&mut self, ctx: &Context<'py>, child: Child<'py>) -> PyResult<Step<'py>> {
        if let Some(child) = child {
            let value = child?;
            match self.kind {
                SequenceKind::List | SequenceKind::Tuple => {
                    self.copy.cast::<PyList>()?.append(value)?
                }
                SequenceKind::Set | SequenceKind::FrozenSet => {
                    self.copy.cast::<PySet>()?.add(value)?
                }
            }
            self.count += 1;
        }
        if let Some(item) = self.items.next().transpose()? {
            if ctx.max_items.is_some_and(|limit| self.count >= limit) {
                ctx.count("max_items")?;
            } else if !ctx.has_nodes()? && !ctx.objects.contains(item.as_ptr() as usize)? {
                ctx.count("max_nodes")?;
            } else {
                return Ok(Step::Value(item, self.depth.saturating_add(1)));
            }
        }
        let constructor = match self.kind {
            SequenceKind::Tuple => PyTuple::type_object(ctx.py),
            SequenceKind::FrozenSet => PyFrozenSet::type_object(ctx.py),
            _ => return Ok(Step::Done(self.copy.clone())),
        };
        let finished = constructor.call1((&self.copy,))?;
        ctx.remember(&self.source, &finished)?;
        Ok(Step::Done(finished))
    }
}
