pub mod config;
pub mod environment;
pub mod grid;
pub mod model;
pub mod osrm;
mod python_api;
pub mod simulator;
pub mod web;

use pyo3::prelude::*;
use python_api::RustSimulation;

/// Register the public Python surface of the compiled `_rust_core` module.
#[pymodule]
fn _rust_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RustSimulation>()?;
    Ok(())
}
