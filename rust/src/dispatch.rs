//! Routes `ValuesMixin.value` and `ValuesMixin._mapping` to a walker.
//!
//! [`Dispatch`] replaces each method on the class. Looked up on a context
//! whose `_walker` is a [`Walker`], it returns itself bound to the context
//! as a method; on any other context, the Python method bound as usual; on
//! the class, the descriptor itself, which routes
//! `SnapshotContext.value(ctx, ...)` and `SnapshotContext.value(self=ctx,
//! ...)` the same way. No Python frame is added, so the Python walk
//! overflows its recursion limit exactly where it would without the walker.
//!
//! The walker takes only the argument shapes [`crate::arguments`] reads.

use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyTypeError};
use pyo3::gc::PyVisit;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};
use pyo3::PyTraverseError;

use crate::walker::{Call, Walker};

// A native method descriptor over a Python walk method. Plain comments,
// not a doc comment: a class docstring would hide the function's own
// `__doc__` from bound methods.
#[pyclass(module = "pydumpling._native", frozen, weakref)]
pub struct Dispatch {
    /// The Python walk's function.
    pure: Py<PyAny>,
    /// Whether this replaces `_mapping` rather than `value`.
    mapping: bool,
    /// The class it is set on, which pickles find it on by name.
    owner: Option<Py<PyAny>>,
    /// `types.MethodType`, which binds it to a context.
    method: Py<PyAny>,
    /// `object.__getattribute__`, which reads `_walker` as stock attribute
    /// access does, without the context class's hooks.
    getattribute: Py<PyAny>,
}

#[pymethods]
impl Dispatch {
    #[new]
    // No generated text signature either: it would be a class docstring.
    #[pyo3(signature = (pure, *, mapping = false, owner = None), text_signature = None)]
    fn new(
        py: Python<'_>,
        pure: Py<PyAny>,
        mapping: bool,
        owner: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let method = py.import("types")?.getattr("MethodType")?.unbind();
        let getattribute = py
            .import("builtins")?
            .getattr("object")?
            .getattr("__getattribute__")?
            .unbind();
        Ok(Self {
            pure,
            mapping,
            owner,
            method,
            getattribute,
        })
    }

    /// The Python walk's function, for introspection.
    #[getter]
    fn __wrapped__(&self, py: Python<'_>) -> Py<PyAny> {
        self.pure.clone_ref(py)
    }

    /// The function's docstring.
    #[getter]
    fn __doc__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.pure.bind(py).getattr(intern!(py, "__doc__"))
    }

    /// Pickles as `getattr(owner, name)`, as the function it replaces does.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(owner) = &self.owner else {
            return Err(PyTypeError::new_err(
                "a Dispatch without an owner cannot be pickled",
            ));
        };
        let getattr = py.import("builtins")?.getattr("getattr")?;
        let name = self.pure.bind(py).getattr(intern!(py, "__name__"))?;
        Ok((getattr, (owner.clone_ref(py), name))
            .into_pyobject(py)?
            .into_any())
    }

    /// Everything else, such as `__name__` and `__qualname__`, is the
    /// function's.
    fn __getattr__<'py>(
        &self,
        py: Python<'py>,
        name: &Bound<'py, PyString>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.pure.bind(py).getattr(name)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.pure)?;
        visit.call(&self.method)?;
        visit.call(&self.getattribute)?;
        if let Some(owner) = &self.owner {
            visit.call(owner)?;
        }
        Ok(())
    }

    fn __get__<'py>(
        slf: &Bound<'py, Self>,
        instance: Option<&Bound<'py, PyAny>>,
        owner: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let Some(instance) = instance.filter(|instance| !instance.is_none()) else {
            return Ok(slf.clone().into_any());
        };
        if slf.get().walker_of(instance)?.is_none() {
            let pure = slf.get().pure.bind(py);
            return pure.call_method1(intern!(py, "__get__"), (instance, owner));
        }
        // A real bound method, as the Python walk's own: calling it calls
        // the descriptor with the context first.
        slf.get().method.bind(py).call1((slf, instance))
    }

    /// `SnapshotContext.value(ctx, ...)`, routed like `ctx.value(...)`.
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if args.is_empty() {
            // The Python function also accepts its receiver by keyword. Route
            // that form through the same walker, without passing `self` twice.
            if let Some(kwargs) = kwargs {
                if let Some(ctx) = kwargs.get_item(intern!(py, "self"))? {
                    let rest = kwargs.copy()?;
                    rest.del_item(intern!(py, "self"))?;
                    return self.route(&ctx, args, Some(&rest));
                }
            }
            return self.pure.bind(py).call(args, kwargs);
        }
        let rest = args.get_slice(1, args.len());
        self.route(&args.get_item(0)?, &rest, kwargs)
    }
}

impl Dispatch {
    fn route<'py>(
        &self,
        ctx: &Bound<'py, PyAny>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = ctx.py();
        let pure = self.pure.bind(py);
        let Some(walker) = self.walker_of(ctx)? else {
            return pure.call(prepend(ctx, args)?, kwargs);
        };
        // A finalizer, collection callback, audit hook or signal handler
        // that runs while this walker captures, or another thread: the
        // call in progress keeps its node count, and what a hand-back would
        // undo, where the Python walk cannot share them.
        let Ok(mut walker) = walker.try_borrow_mut() else {
            return Err(PyRuntimeError::new_err(
                "this SnapshotContext is already capturing: a native context \
                 captures one call at a time, so reentrant or concurrent \
                 capture through it is unsupported",
            ));
        };
        let call = Call {
            pure,
            mapping: self.mapping,
            args,
            kwargs,
        };
        walker.state.run(py, ctx, &call)
    }

    /// The context's walker, if it captures natively, read from its
    /// instance dict or slot as stock attribute access reads it: a class
    /// that overrides `__getattribute__` sees no read the Python walk
    /// would not make.
    fn walker_of<'py>(&self, ctx: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, Walker>>> {
        let py = ctx.py();
        let read = self
            .getattribute
            .bind(py)
            .call1((ctx, intern!(py, "_walker")));
        match read {
            Ok(walker) => Ok(walker.cast_into::<Walker>().ok()),
            Err(err) if err.is_instance_of::<PyAttributeError>(py) => Ok(None),
            Err(err) => Err(err),
        }
    }
}

/// `(ctx, *args)`.
pub(crate) fn prepend<'py>(
    ctx: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyTuple>> {
    let mut items = Vec::with_capacity(args.len() + 1);
    items.push(ctx.clone());
    items.extend(args.iter());
    PyTuple::new(ctx.py(), items)
}
