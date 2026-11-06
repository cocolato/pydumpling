//! Native value walker for pydumpling's bounded snapshots.
//!
//! The pure-Python walk in `pydumpling/snapshot/values.py` is the reference
//! implementation, and [`walker::Walker`] mirrors it step for step: the same
//! memo, node budget, truncation counters, and copies, so a snapshot pickles
//! to the same bytes whichever walk produced it. The walker never runs
//! application code: where the Python walk would, or where its result would
//! depend on its own recursion limit, the walker hands the whole call back
//! to Python before doing anything, so that code runs exactly once.
//!
//! The module is optional and needs Python 3.12+: without it, or with
//! `PYDUMPLING_NATIVE=0`, pydumpling captures with the Python walk alone.
#![forbid(unsafe_code)]

mod arguments;
mod copies;
mod dispatch;
mod keys;
mod mapping;
pub mod memo;
mod mirror;
mod statics;
mod text;
mod transaction;
mod types;
mod walk;
mod walker;

use pyo3::prelude::*;

pub use dispatch::Dispatch;
pub use statics::StaticsHandle;
pub use walker::Walker;

/// The `pydumpling._native` extension module.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<StaticsHandle>()?;
    module.add_class::<Walker>()?;
    module.add_class::<Dispatch>()?;
    Ok(())
}
