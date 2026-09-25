use anyhow::{Context, Result, anyhow};
use pyo3::prelude::*;
use serde::Serialize;

use crate::{
    config::Config,
    grid::DispatchAction,
    model::{Car, Passenger},
};

#[derive(Serialize)]
struct AlgorithmPoint<'a> {
    id: &'a str,
    lat: f64,
    lon: f64,
}

#[derive(Serialize)]
struct MatchingPayload<'a> {
    cars: Vec<AlgorithmPoint<'a>>,
    passengers: Vec<AlgorithmPoint<'a>>,
}

#[derive(Serialize)]
struct ModelPaths {
    ppo: String,
    dqn: String,
}

#[derive(Serialize)]
struct DispatchPayload<'a> {
    observation: &'a [f32],
    model_paths: ModelPaths,
}

#[derive(Clone)]
pub struct PythonAlgorithms {
    algorithm_dir: String,
    matching_name: String,
    dispatch_name: String,
    config: Config,
}

impl PythonAlgorithms {
    pub fn new(config: &Config) -> Self {
        Self {
            algorithm_dir: config
                .resolve(&config.python.algorithm_dir)
                .to_string_lossy()
                .into_owned(),
            matching_name: config.python.matching.clone(),
            dispatch_name: config.python.dispatch.clone(),
            config: config.clone(),
        }
    }

    fn prepare_path<'py>(&self, py: Python<'py>) -> PyResult<()> {
        let sys = py.import("sys")?;
        sys.getattr("path")?
            .call_method1("insert", (0, &self.algorithm_dir))?;
        Ok(())
    }

    pub fn matching(
        &self,
        cars: &[&Car],
        passengers: &[Passenger],
        threshold_m: f64,
    ) -> Result<Vec<(String, String)>> {
        let payload = MatchingPayload {
            cars: cars
                .iter()
                .filter_map(|car| {
                    car.position().map(|p| AlgorithmPoint {
                        id: &car.id,
                        lat: p.lat,
                        lon: p.lon,
                    })
                })
                .collect(),
            passengers: passengers
                .iter()
                .map(|p| AlgorithmPoint {
                    id: &p.id,
                    lat: p.start_lat,
                    lon: p.start_lon,
                })
                .collect(),
        };
        let input = serde_json::to_string(&payload)?;
        Python::with_gil(|py| -> Result<_> {
            self.prepare_path(py)?;
            let module = py.import("api")?;
            let output: String = module
                .getattr("match_json")?
                .call1((input, &self.matching_name, threshold_m))?
                .extract()?;
            serde_json::from_str(&output).context("decode Python matching result")
        })
    }

    pub fn dispatch(&self, observation: &[f32]) -> Result<DispatchAction> {
        let payload = DispatchPayload {
            observation,
            model_paths: ModelPaths {
                ppo: self
                    .config
                    .resolve(&self.config.python.ppo_model)
                    .to_string_lossy()
                    .into_owned(),
                dqn: self
                    .config
                    .resolve(&self.config.python.dqn_model)
                    .to_string_lossy()
                    .into_owned(),
            },
        };
        let input = serde_json::to_string(&payload)?;
        Python::with_gil(|py| -> Result<_> {
            self.prepare_path(py)?;
            let module = py.import("api")?;
            let output: String = module
                .getattr("dispatch_action_json")?
                .call1((input, &self.dispatch_name))?
                .extract()?;
            serde_json::from_str(&output)
                .map_err(|error| anyhow!("decode Python dispatch result: {error}"))
        })
    }
}
