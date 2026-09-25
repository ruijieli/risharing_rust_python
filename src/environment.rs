use std::{net::SocketAddr, path::Path, sync::Arc};

use anyhow::{Result, anyhow};
use tokio::sync::RwLock;

use crate::{
    config::Config,
    grid::{DispatchAction, Grid},
    simulator::Simulator,
    web::{SharedSnapshot, build_snapshot},
};

/// Result of one Gymnasium-style environment transition.
pub struct EnvironmentStep {
    pub observation: Vec<f32>,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub info_json: String,
}

/// Language-neutral RL environment built on top of the simulation kernel.
///
/// PyO3-specific conversion belongs in `python_api.rs`; this type owns the
/// reset/step semantics and can also be used by another Rust frontend later.
pub struct RustEnvironment {
    config: Config,
    simulator: Simulator,
    runtime: tokio::runtime::Runtime,
    snapshot: SharedSnapshot,
    visualization_url: Option<String>,
}

impl RustEnvironment {
    pub fn new(config_path: &Path) -> Result<Self> {
        let config = Config::load(config_path)?;
        let simulator = Simulator::new(config.clone())?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        Ok(Self {
            config,
            simulator,
            runtime,
            snapshot: Arc::new(RwLock::new(None)),
            visualization_url: None,
        })
    }

    pub fn reset(&mut self, seed: Option<u64>) -> Result<Vec<f32>> {
        if let Some(seed) = seed {
            self.config.simulation.random_seed = seed;
        }
        self.simulator = Simulator::new(self.config.clone())?;
        self.publish_snapshot();
        Ok(self.observation())
    }

    pub fn observation(&self) -> Vec<f32> {
        self.simulator.observation()
    }

    pub fn num_grids(&self) -> usize {
        self.simulator.num_grids()
    }

    pub fn grid_cell_ids(&self) -> Vec<String> {
        self.simulator.grid_cell_ids()
    }

    pub fn step(&mut self, action: DispatchAction) -> Result<EnvironmentStep> {
        let reward = self
            .runtime
            .block_on(self.simulator.training_cycle(action))?;
        self.publish_snapshot();
        Ok(EnvironmentStep {
            observation: self.observation(),
            reward,
            terminated: self.simulator.done(),
            truncated: false,
            info_json: serde_json::json!({
                "served_passengers": reward,
                "current_time": self.simulator.current_time,
            })
            .to_string(),
        })
    }

    pub fn start_visualization(&mut self) -> Result<String> {
        if let Some(url) = &self.visualization_url {
            self.publish_snapshot();
            return Ok(url.clone());
        }

        let mut config = self.config.visualization.clone();
        let requested_port = config.port;
        let mut bound = None;
        for port in requested_port..=requested_port.saturating_add(20) {
            let address: SocketAddr = format!("{}:{port}", config.host).parse()?;
            match self
                .runtime
                .block_on(tokio::net::TcpListener::bind(address))
            {
                Ok(listener) => {
                    bound = Some((port, listener));
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => continue,
                Err(error) => return Err(error.into()),
            }
        }
        let (port, listener) = bound.ok_or_else(|| {
            anyhow!(
                "visualization ports {requested_port}-{} are all in use",
                requested_port.saturating_add(20)
            )
        })?;
        config.port = port;
        let url = format!("http://{}:{port}", config.host);
        let snapshot = self.snapshot.clone();
        let grid = Grid::new(&self.config)?.geojson();
        self.runtime.spawn(async move {
            if let Err(error) =
                crate::web::serve_with_listener(config, snapshot, grid, listener).await
            {
                eprintln!("Visualization server stopped: {error}");
            }
        });
        self.visualization_url = Some(url.clone());
        self.publish_snapshot();
        Ok(url)
    }

    fn publish_snapshot(&self) {
        let value = build_snapshot(&self.simulator.waiting, &self.simulator.cars);
        self.runtime.block_on(async {
            *self.snapshot.write().await = Some(value);
        });
    }
}
