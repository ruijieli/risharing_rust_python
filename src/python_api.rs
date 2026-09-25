use std::path::Path;

use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{
    environment::{EnvironmentStep, RustEnvironment},
    grid::DispatchAction,
};

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

    fn step_continuous(&mut self, action: Vec<Vec<f64>>) -> PyResult<PythonStep> {
        self.step_action(DispatchAction::Matrix { values: action })
    }

    fn step_discrete(&mut self, action: usize) -> PyResult<PythonStep> {
        self.step_action(DispatchAction::Discrete { value: action })
    }
}

impl RustSimulation {
    fn step_action(&mut self, action: DispatchAction) -> PyResult<PythonStep> {
        self.environment
            .step(action)
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
