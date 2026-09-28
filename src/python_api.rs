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

    fn total_passengers(&self) -> usize {
        self.environment.total_passengers()
    }

    fn served_passengers(&self) -> u64 {
        self.environment.served_passengers()
    }

    fn grid_cell_ids(&self) -> Vec<String> {
        self.environment.grid_cell_ids()
    }

    fn start_visualization(&mut self) -> PyResult<String> {
        self.environment.start_visualization().map_err(py_error)
    }

    fn matching_input_json(&mut self) -> PyResult<String> {
        self.environment.matching_input_json().map_err(py_error)
    }

    #[pyo3(signature = (matches, counts=None))]
    fn step(
        &mut self,
        matches: Vec<(String, String)>,
        counts: Option<Vec<Vec<usize>>>,
    ) -> PyResult<PythonStep> {
        self.environment
            .step(
                matches,
                counts.map(|counts| crate::grid::DispatchAction::FlowMatrix { counts }),
            )
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

fn py_error(error: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{error:#}"))
}
