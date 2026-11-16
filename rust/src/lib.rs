//! Native value walker for pydumpling's bounded snapshots.
//!
//! The pure-Python walk in `pydumpling/snapshot/values.py` is the reference
//! implementation, and the walker mirrors it step for step, on the
//! context's own memo, node budget and truncation counters, so wherever the
//! Python walk stays clear of its recursion limit a snapshot pickles to the
//! same bytes whichever walk produced it, and application code runs at the
//! same steps the same number of times. `ValuesMixin.value` and `_mapping`
//! call the walker's entries ([`walker`]) on a context whose class was
//! found, when it was created, to run the stock walk; an entry returns the
//! copy, or `_IN_PYTHON` before the call has done anything, and the method
//! then runs the call itself in its own frame. Once a call has started, the
//! walker runs application code where the Python walk runs it, through the
//! same C entry points, and runs a node it does not model, or one below its
//! native stack budget, in the Python walk in place. It calls none of
//! pydumpling's Python helpers, so tracing and profiling hooks see only
//! that application code, those nodes, and `FakeContainer.__hash__` where
//! a set or dict hashes a container copy, through C as in the Python walk.
//! CPython's C recursion limit is excepted for the code and the nodes,
//! which run beneath the walker's C frames (see the contract and "The C
//! recursion limit" in `rust/DESIGN.md`).
//!
//! The module is optional and needs CPython 3.12+ with the GIL. Where it is
//! not loaded, under `python -b`, or with `PYDUMPLING_NATIVE=0`, pydumpling
//! captures with the Python walk alone.
#![forbid(unsafe_code)]

mod address;
mod copies;
mod keys;
mod layouts;
mod mapping;
mod objects;
mod state;
mod statics;
mod text;
mod types;
mod walk;
mod walker;

use pyo3::prelude::*;

pub use walker::StaticsHandle;

/// The `pydumpling._native` extension module.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<StaticsHandle>()?;
    // What `rust/build_native.py` built this from: see its `fingerprint`.
    module.add("fingerprint", option_env!("PYDUMPLING_NATIVE_FINGERPRINT"))?;
    Ok(())
}
