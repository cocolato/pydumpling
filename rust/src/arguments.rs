//! Reads a call's arguments in the shapes the Python walk's own callers
//! pass them: exact ints, bools, and containers of plain keys, so no
//! application code runs before the walk starts. Any other call runs in the
//! Python walk as it was made, so Python raises or converts as it would.

use pyo3::prelude::*;
use pyo3::types::{PyFrozenSet, PyList, PySet, PyString, PyTuple};

use crate::statics::{type_address, Statics};
use crate::walker::Call;

/// Depths the walker takes; anything larger runs in Python, whose ints
/// never overflow.
const DEPTH_LIMIT: i64 = 1 << 62;

/// A call's arguments in the shapes the walker takes.
pub(crate) enum Parsed<'py> {
    Value {
        value: Bound<'py, PyAny>,
        depth: i64,
    },
    Mapping {
        source: Bound<'py, PyAny>,
        depth: i64,
        priority: Option<Bound<'py, PyAny>>,
        counted: bool,
        container: Bound<'py, PyAny>,
    },
}

/// Reads `call`'s arguments, or `None` if the Python walk must take them:
/// only exact ints, bools, and containers of plain keys are read natively,
/// so no application code runs before the walk starts.
pub(crate) fn parse<'py>(s: &Statics, call: &Call<'_, 'py>) -> PyResult<Option<Parsed<'py>>> {
    let args = call.args;
    if args.is_empty() || args.len() > 2 {
        return Ok(None);
    }
    let first = args.get_item(0)?;
    let mut depth = match args.len() {
        2 => match exact_depth(s, &args.get_item(1)?) {
            Some(depth) => Some(depth),
            None => return Ok(None),
        },
        _ => None,
    };
    let positional_depth = depth.is_some();
    let mut priority = None;
    let mut counted = false;
    let mut container = None;
    if let Some(kwargs) = call.kwargs {
        for (key, item) in kwargs.iter() {
            if type_address(&key) != s.str_ {
                return Ok(None);
            }
            let key = key.cast::<PyString>().map_err(PyErr::from)?;
            // A lone surrogate names no parameter: Python raises for it.
            let Ok(name) = key.to_str() else {
                return Ok(None);
            };
            match (name, call.mapping) {
                ("depth", _) if !positional_depth => match exact_depth(s, &item) {
                    Some(value) => depth = Some(value),
                    None => return Ok(None),
                },
                ("priority", true) if plain_keys(s, &item)? => priority = Some(item),
                ("counted", true) if item.is(s.true_.bind(item.py())) => counted = true,
                ("counted", true) if item.is(s.false_.bind(item.py())) => counted = false,
                ("container", true)
                    if item.is(s.dict_cls.bind(item.py()))
                        || item.is(s.fake_dict.bind(item.py())) =>
                {
                    container = Some(item)
                }
                _ => return Ok(None),
            }
        }
    }
    let depth = depth.unwrap_or(0);
    Ok(Some(if call.mapping {
        let py = first.py();
        Parsed::Mapping {
            source: first,
            depth,
            priority,
            counted,
            container: container.unwrap_or_else(|| s.dict_cls.bind(py).clone()),
        }
    } else {
        Parsed::Value {
            value: first,
            depth,
        }
    }))
}

/// An exact int the walk can add to without overflow.
fn exact_depth(s: &Statics, value: &Bound<'_, PyAny>) -> Option<i64> {
    if type_address(value) != s.int {
        return None;
    }
    value
        .extract::<i64>()
        .ok()
        .filter(|depth| depth.abs() <= DEPTH_LIMIT)
}

/// Whether `priority` is an exact tuple, list, set, or frozenset of keys
/// whose hashing and comparison run only builtin code.
fn plain_keys(s: &Statics, priority: &Bound<'_, PyAny>) -> PyResult<bool> {
    let ty = type_address(priority);
    if ty == s.tuple {
        let items = priority.cast::<PyTuple>().map_err(PyErr::from)?;
        return Ok(items.iter().all(|key| s.plain_key(&key)));
    }
    if ty == s.list {
        let items = priority.cast::<PyList>().map_err(PyErr::from)?;
        return Ok(items.iter().all(|key| s.plain_key(&key)));
    }
    if ty == s.set {
        let items = priority.cast::<PySet>().map_err(PyErr::from)?;
        return Ok(items.iter().all(|key| s.plain_key(&key)));
    }
    if ty == s.frozenset {
        let items = priority.cast::<PyFrozenSet>().map_err(PyErr::from)?;
        return Ok(items.iter().all(|key| s.plain_key(&key)));
    }
    Ok(false)
}
