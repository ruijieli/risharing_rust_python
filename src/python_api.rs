use std::path::Path;

use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::environment::{EnvironmentStep, RustEnvironment};

type PythonStep = (Vec<f32>, f64, bool, bool, String);

/// Thin PyO3 adapter. Environment semantics live in `environment.rs`.
#[pyclass(unsendable)]
pub(crate) struct RustSimulation {
    environment: RustEnvironment,
}

#[pymethods]
impl RustSimulation {
    #[new]
    fn new(config_path: String) -> PyResult<Self> {
        Ok(Self {
            environment: RustEnvironment::new(Path::new(&config_path)).map_err(py_error)?,
        })
    }

    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, seed: Option<u64>) -> PyResult<Vec<f32>> {
        self.environment.reset(seed).map_err(py_error)
    }

    fn observation(&self) -> Vec<f32> {
        self.environment.observation()
    }

    fn num_grids(&self) -> usize {
        self.environment.num_grids()
    }

    fn grid_cell_ids(&self) -> Vec<String> {
        self.environment.grid_cell_ids()
    }

    fn start_visualization(&mut self) -> PyResult<String> {
        self.environment.start_visualization().map_err(py_error)
    }

    fn step_flow_matrix(&mut self, counts: Vec<Vec<usize>>) -> PyResult<PythonStep> {
        self.environment
            .step_flow_matrix(counts)
            .map(environment_step_to_python)
            .map_err(py_error)
    }

    fn step_none(&mut self) -> PyResult<PythonStep> {
        self.environment
            .step_none()
            .map(environment_step_to_python)
            .map_err(py_error)
    }
}

fn environment_step_to_python(step: EnvironmentStep) -> PythonStep {
    (
        step.observation,
        step.reward,
        step.terminated,
        step.truncated,
        step.info_json,
    )
}

fn py_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}
