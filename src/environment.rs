use std::{net::SocketAddr, path::Path, sync::Arc};

use anyhow::{Result, anyhow};
use tokio::sync::RwLock;

use crate::{
    config::Config,
    grid::Grid,
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

    pub fn total_passengers(&self) -> usize {
        self.simulator.passengers.len()
    }

    pub fn served_passengers(&self) -> u64 {
        self.simulator.total_served
    }

    pub fn grid_cell_ids(&self) -> Vec<String> {
        self.simulator.grid_cell_ids()
    }

    pub fn step(&mut self, matches: Vec<(String, String)>) -> Result<EnvironmentStep> {
        let reward = self
            .runtime
            .block_on(self.simulator.advance_step(matches))?;
        self.environment_step(reward)
    }

    pub fn apply_matches(&mut self, matches: Vec<(String, String)>) -> Result<f64> {
        let reward = self
            .runtime
            .block_on(self.simulator.apply_matches(matches))?;
        self.publish_snapshot();
        Ok(reward)
    }

    pub fn advance(&mut self, proportions: Vec<Vec<f64>>) -> Result<EnvironmentStep> {
        self.runtime
            .block_on(self.simulator.advance(Some(&proportions)))?;
        self.environment_step(0.0)
    }

    fn environment_step(&mut self, reward: f64) -> Result<EnvironmentStep> {
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

    pub fn matching_input_json(&mut self) -> Result<String> {
        Ok(serde_json::to_string(&self.simulator.matching_input())?)
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
