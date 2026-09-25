pub mod config;
pub mod grid;
pub mod model;
pub mod osrm;
pub mod python_bridge;
pub mod simulator;
pub mod web;

use std::{net::SocketAddr, path::Path, sync::Arc};

use pyo3::{exceptions::PyRuntimeError, prelude::*};

use tokio::sync::RwLock;

use crate::{
    config::Config,
    grid::{DispatchAction, Grid},
    simulator::Simulator,
    web::{SharedSnapshot, build_snapshot},
};

/// Python/Gymnasium-facing wrapper around the same Rust simulation kernel.
#[pyclass(unsendable)]
struct RustSimulation {
    config: Config,
    simulator: Simulator,
    runtime: tokio::runtime::Runtime,
    snapshot: SharedSnapshot,
    visualization_started: bool,
    visualization_url: Option<String>,
}

#[pymethods]
impl RustSimulation {
    #[new]
    fn new(config_path: String) -> PyResult<Self> {
        let config = Config::load(Path::new(&config_path)).map_err(py_error)?;
        let simulator = Simulator::new(config.clone()).map_err(py_error)?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(py_error)?;
        Ok(Self {
            config,
            simulator,
            runtime,
            snapshot: Arc::new(RwLock::new(None)),
            visualization_started: false,
            visualization_url: None,
        })
    }

    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, seed: Option<u64>) -> PyResult<Vec<f32>> {
        if let Some(seed) = seed {
            self.config.simulation.random_seed = seed;
        }
        self.simulator = Simulator::new(self.config.clone()).map_err(py_error)?;
        self.publish_snapshot();
        Ok(self.simulator.observation())
    }

    fn observation(&self) -> Vec<f32> {
        self.simulator.observation()
    }
    fn num_grids(&self) -> usize {
        self.simulator.num_grids()
    }
    fn grid_cell_ids(&self) -> Vec<String> {
        self.simulator.grid_cell_ids()
    }

    fn start_visualization(&mut self) -> PyResult<String> {
        if let Some(url) = &self.visualization_url {
            self.publish_snapshot();
            return Ok(url.clone());
        }
        if !self.visualization_started {
            let mut config = self.config.visualization.clone();
            let snapshot = self.snapshot.clone();
            let grid = Grid::new(&self.config).map_err(py_error)?.geojson();
            let requested_port = config.port;
            let mut bound = None;
            for port in requested_port..=requested_port.saturating_add(20) {
                let address: SocketAddr = format!("{}:{port}", config.host)
                    .parse()
                    .map_err(py_error)?;
                match self
                    .runtime
                    .block_on(tokio::net::TcpListener::bind(address))
                {
                    Ok(listener) => {
                        bound = Some((port, listener));
                        break;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => continue,
                    Err(error) => return Err(py_error(error)),
                }
            }
            let (port, listener) = bound.ok_or_else(|| {
                py_error(format!(
                    "visualization ports {requested_port}-{} are all in use",
                    requested_port.saturating_add(20)
                ))
            })?;
            config.port = port;
            let url = format!("http://{}:{port}", config.host);
            self.runtime.spawn(async move {
                if let Err(error) =
                    crate::web::serve_with_listener(config, snapshot, grid, listener).await
                {
                    eprintln!("Visualization server stopped: {error}");
                }
            });
            self.visualization_started = true;
            self.visualization_url = Some(url);
        }
        self.publish_snapshot();
        Ok(self
            .visualization_url
            .clone()
            .expect("visualization URL set"))
    }

    fn step_continuous(
        &mut self,
        action: Vec<Vec<f64>>,
    ) -> PyResult<(Vec<f32>, f64, bool, bool, String)> {
        self.step_action(DispatchAction::Matrix { values: action })
    }

    fn step_discrete(&mut self, action: usize) -> PyResult<(Vec<f32>, f64, bool, bool, String)> {
        self.step_action(DispatchAction::Discrete { value: action })
    }
}

impl RustSimulation {
    fn step_action(
        &mut self,
        action: DispatchAction,
    ) -> PyResult<(Vec<f32>, f64, bool, bool, String)> {
        let reward = self
            .runtime
            .block_on(self.simulator.training_cycle(action))
            .map_err(py_error)?;
        self.publish_snapshot();
        let info = serde_json::json!({
            "served_passengers": reward,
            "current_time": self.simulator.current_time,
        })
        .to_string();
        Ok((
            self.simulator.observation(),
            reward,
            self.simulator.done(),
            false,
            info,
        ))
    }

    fn publish_snapshot(&self) {
        let value = build_snapshot(&self.simulator.waiting, &self.simulator.cars);
        self.runtime.block_on(async {
            *self.snapshot.write().await = Some(value);
        });
    }
}

fn py_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

#[pymodule]
fn _rust_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RustSimulation>()?;
    Ok(())
}
