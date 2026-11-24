//! Suspend mapping capture while the scheduler converts a key or value.

use pyo3::exceptions::{PyRecursionError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PySet, PyString, PyTuple};

use crate::{Context, Obj, Step};

struct Selection<'py> {
    entries: Bound<'py, PyIterator>,
    limit: Option<usize>,
    reason: String,
    count: usize,
    names: bool,
}

enum Waiting<'py> {
    Next,
    Key {
        item: Obj<'py>,
    },
    Value {
        key: Obj<'py>,
        item: Obj<'py>,
        mark: Option<usize>,
    },
}

pub(crate) struct Mapping<'py> {
    captured: Obj<'py>,
    depth: i64,
    selection: Option<Selection<'py>>,
    waiting: Waiting<'py>,
}

impl<'py> Mapping<'py> {
    pub(crate) fn new(
        ctx: &Context<'py>,
        source: Obj<'py>,
        depth: i64,
        priority: Obj<'py>,
        counted: bool,
        container: Obj<'py>,
    ) -> PyResult<Self> {
        if let Some(cached) = ctx.memo(&source)? {
            if cached.is_exact_instance_of::<PyDict>() {
                return Ok(Self::finished(cached, depth));
            }
            let types = ctx.py.import("pydumpling.snapshot.values")?;
            if cached.get_type().is(&types.getattr("FakeDict")?) {
                return Ok(Self::finished(cached, depth));
            }
            if !cached.is_instance(&ctx.truncated)? || cached.getattr("reason")?.ne("max_depth")? {
                ctx.count("unsupported_type")?;
                return Ok(Self::finished(PyDict::new(ctx.py).into_any(), depth));
            }
        }
        if !counted {
            if !ctx.has_nodes()? {
                ctx.count("max_nodes")?;
                return Ok(Self::finished(PyDict::new(ctx.py).into_any(), depth));
            }
            ctx.charge()?;
        }
        let max_depth = ctx.depth_reached(depth);
        if max_depth {
            ctx.count("max_depth")?;
        }
        let captured = container.call0()?;
        ctx.remember(&source, &captured)?;
        let mut frame = Self::finished(captured, depth);
        if max_depth {
            return Ok(frame);
        }
        let names = ctx
            .ctx
            .getattr("_namespaces")?
            .contains(source.as_ptr() as usize)?;
        // Converting priority precedes the mutation handler in the reference.
        let preferred = ctx.py.get_type::<PySet>().call1((priority,))?;
        match ctx
            .ctx
            .call_method1("_mapping_items", (source, names, preferred))
        {
            Ok(selected) => {
                let (items, limit, reason): (Obj<'py>, Option<usize>, String) =
                    selected.extract()?;
                frame.selection = Some(Selection {
                    entries: items.try_iter()?,
                    limit,
                    reason,
                    count: 0,
                    names,
                });
            }
            Err(err) if is_mutation(ctx.py, &err) => ctx.count("concurrent_mutation")?,
            Err(err) => return Err(err),
        }
        Ok(frame)
    }

    fn finished(captured: Obj<'py>, depth: i64) -> Self {
        Self {
            captured,
            depth,
            selection: None,
            waiting: Waiting::Next,
        }
    }

    pub(crate) fn resume(
        &mut self,
        ctx: &Context<'py>,
        child: Option<PyResult<Obj<'py>>>,
    ) -> PyResult<Step<'py>> {
        match self.advance(ctx, child) {
            Err(err) if is_mutation(ctx.py, &err) => {
                ctx.count("concurrent_mutation")?;
                Ok(Step::Done(self.captured.clone()))
            }
            outcome => outcome,
        }
    }

    fn advance(
        &mut self,
        ctx: &Context<'py>,
        child: Option<PyResult<Obj<'py>>>,
    ) -> PyResult<Step<'py>> {
        match std::mem::replace(&mut self.waiting, Waiting::Next) {
            Waiting::Key { item } => {
                return self.start_value(ctx, child.expect("mapping key result")?, item);
            }
            Waiting::Value { key, item, mark } => {
                let value = match child.expect("mapping value result") {
                    Err(err)
                        if mark.is_some() && err.is_instance_of::<PyRecursionError>(ctx.py) =>
                    {
                        ctx.ctx.call_method1("_recover_root", (item, mark))?
                    }
                    outcome => outcome?,
                };
                if let Err(err) = self.captured.set_item(&key, &value) {
                    if !err.is_instance_of::<PyTypeError>(ctx.py) {
                        return Err(err);
                    }
                    ctx.count("unsupported_key")?;
                }
            }
            Waiting::Next => {}
        }
        let Some(selection) = self.selection.as_mut() else {
            return Ok(Step::Done(self.captured.clone()));
        };
        let Some(entry) = selection.entries.next() else {
            return Ok(Step::Done(self.captured.clone()));
        };
        let entry = entry?.cast_into::<PyTuple>()?;
        if selection
            .limit
            .is_some_and(|limit| selection.count >= limit)
        {
            ctx.count(&selection.reason)?;
            return Ok(Step::Done(self.captured.clone()));
        }
        if !ctx.has_nodes()? {
            ctx.count("max_nodes")?;
            return Ok(Step::Done(self.captured.clone()));
        }
        let key = entry.get_item(0)?;
        let item = entry.get_item(1)?;
        selection.count += 1;
        if selection.names && key.is_exact_instance_of::<PyString>() {
            let key = ctx.ctx.call_method1("_code_text", (key,))?;
            self.start_value(ctx, key, item)
        } else {
            self.waiting = Waiting::Key { item };
            Ok(Step::Value(key, self.depth.saturating_add(1)))
        }
    }

    fn start_value(
        &mut self,
        ctx: &Context<'py>,
        key: Obj<'py>,
        item: Obj<'py>,
    ) -> PyResult<Step<'py>> {
        let mark = if self.depth == 0 {
            Some(ctx.ctx.getattr("objects")?.len()?)
        } else {
            None
        };
        self.waiting = Waiting::Value {
            key,
            item: item.clone(),
            mark,
        };
        Ok(Step::Value(item, self.depth.saturating_add(1)))
    }
}

fn is_mutation(py: Python<'_>, err: &PyErr) -> bool {
    err.is_instance_of::<PyRuntimeError>(py) && !err.is_instance_of::<PyRecursionError>(py)
}
