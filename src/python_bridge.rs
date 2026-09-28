use anyhow::{Context, Result, anyhow};
use pyo3::prelude::*;
use serde::Serialize;
use serde_json::{Map, Value};

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
struct DispatchPayload<'a> {
    observation: &'a [f32],
    options: &'a Map<String, Value>,
    project_dir: &'a str,
}

#[derive(Clone)]
pub struct PythonAlgorithms {
    algorithm_dir: String,
    project_dir: String,
    matching_name: String,
    matching_options: Map<String, Value>,
    dispatch_name: String,
    dispatch_options: Map<String, Value>,
}

impl PythonAlgorithms {
    pub fn new(config: &Config) -> Self {
        Self {
            algorithm_dir: config
                .resolve(&config.python.algorithm_dir)
                .to_string_lossy()
                .into_owned(),
            project_dir: config.project_dir.to_string_lossy().into_owned(),
            matching_name: config.python.matching.name.clone(),
            matching_options: config.python.matching.options.clone(),
            dispatch_name: config.python.dispatch.name.clone(),
            dispatch_options: config.python.dispatch.options.clone(),
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
        let input = serde_json::to_string(&serde_json::json!({
            "cars": payload.cars,
            "passengers": payload.passengers,
            "threshold_m": threshold_m,
            "options": &self.matching_options,
            "project_dir": &self.project_dir,
        }))?;
        Python::with_gil(|py| -> Result<_> {
            self.prepare_path(py)?;
            let module = py.import("api")?;
            let output: String = module
                .getattr("match_json")?
                .call1((input, &self.matching_name))?
                .extract()?;
            serde_json::from_str(&output).context("decode Python matching result")
        })
    }

    pub fn dispatch(&self, observation: &[f32]) -> Result<DispatchAction> {
        let payload = DispatchPayload {
            observation,
            options: &self.dispatch_options,
            project_dir: &self.project_dir,
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
