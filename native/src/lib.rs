#![forbid(unsafe_code)]

mod mapping;
mod sequence;

use mapping::Mapping;
use pyo3::exceptions::{PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyByteArray, PyBytes, PyComplex, PyDict, PyFloat, PyInt, PyString, PyTuple,
};
use pyo3::{intern, PyTypeInfo};
use sequence::{Sequence, SequenceKind};

type Obj<'py> = Bound<'py, PyAny>;
type Child<'py> = Option<PyResult<Obj<'py>>>;

enum Step<'py> {
    Done(Obj<'py>),
    Value(Obj<'py>, i64),
    Mapping(Obj<'py>, i64, Obj<'py>, bool, Obj<'py>),
}

struct Context<'py> {
    py: Python<'py>,
    ctx: Obj<'py>,
    objects: Bound<'py, PyDict>,
    sources: Bound<'py, PyDict>,
    truncation: Obj<'py>,
    max_nodes: Obj<'py>,
    max_depth: Option<u64>,
    max_items: Option<u64>,
    max_string_bytes: Obj<'py>,
    string_limit: Option<u64>,
    truncated: Obj<'py>,
    recursive_reason: Obj<'py>,
    custom_value: Obj<'py>,
}

fn limit(value: &Obj<'_>) -> PyResult<Option<u64>> {
    if value.is_none() {
        return Ok(None);
    }
    // DumpOptions validates positive exact Python integers, including those
    // larger than any container can hold.
    match value.extract() {
        Ok(limit) => Ok(Some(limit)),
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyOverflowError>(value.py()) => {
            Ok(Some(u64::MAX))
        }
        Err(err) => Err(err),
    }
}

impl<'py> Context<'py> {
    fn new(ctx: Obj<'py>) -> PyResult<Self> {
        let py = ctx.py();
        let options = ctx.getattr(intern!(py, "options"))?;
        let max_string_bytes = options.getattr(intern!(py, "max_string_bytes"))?;
        let values = py.import("pydumpling.snapshot.values")?;
        Ok(Self {
            py,
            objects: ctx.getattr(intern!(py, "objects"))?.cast_into()?,
            sources: ctx.getattr(intern!(py, "_sources"))?.cast_into()?,
            truncation: ctx.getattr(intern!(py, "_truncation"))?,
            max_nodes: options.getattr(intern!(py, "max_nodes"))?,
            max_depth: limit(&options.getattr(intern!(py, "max_depth"))?)?,
            max_items: limit(&options.getattr(intern!(py, "max_items"))?)?,
            string_limit: limit(&max_string_bytes)?,
            max_string_bytes,
            truncated: values.getattr(intern!(py, "TruncatedValue"))?,
            recursive_reason: values.getattr(intern!(py, "_RECURSIVE_IMMUTABLE"))?,
            custom_value: py
                .import("pydumpling.snapshot._native_values")?
                .getattr(intern!(py, "custom_value"))?,
            ctx,
        })
    }

    fn memo(&self, source: &Obj<'py>) -> PyResult<Option<Obj<'py>>> {
        self.objects.get_item(source.as_ptr() as usize)
    }

    fn remember(&self, source: &Obj<'py>, copy: &Obj<'py>) -> PyResult<()> {
        // Separate integer keys match Python's two calls to id(original).
        let id = source.as_ptr() as usize;
        self.objects.set_item(id, copy)?;
        self.sources.set_item(id, source)
    }

    fn has_nodes(&self) -> PyResult<bool> {
        Ok(self.max_nodes.is_none()
            || self
                .ctx
                .getattr(intern!(self.py, "_nodes"))?
                .lt(&self.max_nodes)?)
    }

    fn charge(&self) -> PyResult<()> {
        let name = intern!(self.py, "_nodes");
        self.ctx.setattr(name, self.ctx.getattr(name)?.add(1)?)
    }

    fn count(&self, reason: &str) -> PyResult<()> {
        let reason = PyString::intern(self.py, reason);
        let count = self.truncation.get_item(&reason)?.add(1)?;
        self.truncation.set_item(reason, count)
    }

    fn depth_reached(&self, depth: i64) -> bool {
        depth >= 0 && self.max_depth.is_some_and(|limit| depth as u64 >= limit)
    }

    fn value(&self, value: Obj<'py>, depth: i64) -> PyResult<Dispatch<'py>> {
        let py = self.py;
        if let Some(copy) = self.memo(&value)? {
            return Ok(Dispatch::Done(copy));
        }
        if !self.has_nodes()? {
            self.count("max_nodes")?;
            return Ok(Dispatch::Done(
                self.ctx.getattr(intern!(py, "_node_marker"))?,
            ));
        }
        self.charge()?;
        let captured = if value.is_none()
            || value.is_exact_instance_of::<PyBool>()
            || value.is_exact_instance_of::<PyFloat>()
            || value.is_exact_instance_of::<PyComplex>()
            || value.is_exact_instance(&self.truncated)
        {
            value.clone()
        } else if value.is_exact_instance_of::<PyInt>() {
            let too_long = match self.string_limit {
                Some(limit) => {
                    let bits: u64 = value.call_method0(intern!(py, "bit_length"))?.extract()?;
                    bits > limit.saturating_mul(8)
                }
                None => false,
            };
            if too_long {
                self.ctx.call_method1(
                    intern!(py, "_omit"),
                    (intern!(py, "max_string_bytes"), &value),
                )?
            } else {
                value.clone()
            }
        } else if value.is_exact_instance_of::<PyString>() {
            self.ctx.call_method1(intern!(py, "_text"), (&value,))?
        } else if value.is_exact_instance_of::<PyBytes>()
            || value.is_exact_instance_of::<PyByteArray>()
        {
            self.ctx.call_method1(
                intern!(py, "_bounded_bytes"),
                (
                    &value,
                    &self.max_string_bytes,
                    intern!(py, "max_string_bytes"),
                ),
            )?
        } else {
            let sequence = SequenceKind::of(&value);
            let dict = value.is_exact_instance_of::<PyDict>();
            if sequence.is_none() && !dict {
                let program = self.custom_value.call1((&self.ctx, value, depth))?;
                return Ok(Dispatch::Frame(Frame::Generator(program)));
            }
            if self.depth_reached(depth) {
                self.ctx
                    .call_method1(intern!(py, "_omit"), (intern!(py, "max_depth"), &value))?
            } else if let Some(kind) = sequence {
                return Ok(Dispatch::Frame(Frame::Sequence(Sequence::new(
                    self, value, depth, kind,
                )?)));
            } else {
                let mapping = Mapping::new(
                    self,
                    value,
                    depth,
                    PyTuple::empty(py).into_any(),
                    true,
                    PyDict::type_object(py).into_any(),
                )?;
                return Ok(Dispatch::Frame(Frame::Mapping(mapping)));
            }
        };
        self.remember(&value, &captured)?;
        Ok(Dispatch::Done(captured))
    }
}

enum Frame<'py> {
    Generator(Obj<'py>),
    Sequence(Sequence<'py>),
    Mapping(Mapping<'py>),
}

enum Dispatch<'py> {
    Done(Obj<'py>),
    Frame(Frame<'py>),
}

fn resume_generator<'py>(program: &Obj<'py>, child: Child<'py>) -> PyResult<Step<'py>> {
    let py = program.py();
    let result = match child {
        Some(Err(err)) => program.call_method1(intern!(py, "throw"), (err.into_value(py),)),
        Some(Ok(value)) => program.call_method1(intern!(py, "send"), (value,)),
        None => program.call_method1(intern!(py, "send"), (py.None(),)),
    };
    let request = match result {
        Ok(request) => request.cast_into::<PyTuple>()?,
        Err(err) if err.is_instance_of::<PyStopIteration>(py) => {
            return Ok(Step::Done(err.value(py).getattr(intern!(py, "value"))?));
        }
        Err(err) => return Err(err),
    };
    let name = request.get_item(0)?;
    let source = request.get_item(1)?;
    if name.eq(intern!(py, "raise"))? && request.len() == 2 {
        return Err(PyErr::from_value(source));
    }
    let depth = request.get_item(2)?.extract()?;
    if name.eq(intern!(py, "value"))? && request.len() == 3 {
        return Ok(Step::Value(source, depth));
    }
    if name.eq(intern!(py, "mapping"))? && request.len() == 6 {
        return Ok(Step::Mapping(
            source,
            depth,
            request.get_item(3)?,
            request.get_item(4)?.extract()?,
            request.get_item(5)?,
        ));
    }
    Err(PyValueError::new_err("unknown snapshot traversal request"))
}

/// Drive recursive snapshot operations without retaining a Rust or Python
/// call frame for each edge of the input graph.
#[pyfunction]
fn capture<'py>(ctx: Obj<'py>, program: Obj<'py>) -> PyResult<Obj<'py>> {
    let ctx = Context::new(ctx)?;
    let mut frames = vec![Frame::Generator(program)];
    let mut child = None;
    while let Some(frame) = frames.last_mut() {
        let result = match frame {
            Frame::Generator(program) => resume_generator(program, child.take()),
            Frame::Sequence(sequence) => sequence.resume(&ctx, child.take()),
            Frame::Mapping(mapping) => mapping.resume(&ctx, child.take()),
        };
        let next = match result {
            Ok(Step::Done(value)) => {
                frames.pop();
                child = Some(Ok(value));
                continue;
            }
            Err(err) => {
                frames.pop();
                child = Some(Err(err));
                continue;
            }
            Ok(Step::Value(source, depth)) => ctx.value(source, depth),
            Ok(Step::Mapping(source, depth, priority, counted, container)) => {
                Mapping::new(&ctx, source, depth, priority, counted, container)
                    .map(|frame| Dispatch::Frame(Frame::Mapping(frame)))
            }
        };
        match next {
            Ok(Dispatch::Frame(frame)) => frames.push(frame),
            Ok(Dispatch::Done(value)) => child = Some(Ok(value)),
            Err(err) => child = Some(Err(err)),
        }
    }
    child.expect("the root program completed")
}

#[pymodule]
fn pydumpling_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(capture, module)?)?;
    Ok(())
}
