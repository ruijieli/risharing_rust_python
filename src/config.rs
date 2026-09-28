use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub geo: GeoConfig,
    pub simulation: SimulationConfig,
    pub osrm: OsrmConfig,
    pub python: PythonConfig,
    pub visualization: VisualizationConfig,
    #[serde(skip)]
    pub project_dir: PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
pub struct GeoConfig {
    pub max_lat: f64,
    pub max_lon: f64,
    pub min_lat: f64,
    pub min_lon: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SimulationConfig {
    pub passenger_file: String,
    pub num_cars: usize,
    pub distance_threshold_m: f64,
    pub position_update_interval_s: u64,
    pub start_time_s: u64,
    pub end_time_s: u64,
    pub batch_interval_s: u64,
    pub dispatch_cycle_s: u64,
    pub sample_size: Option<usize>,
    pub random_seed: u64,
    pub visualization: bool,
    pub visualization_step_delay_ms: u64,
    pub keep_visualization_alive: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OsrmConfig {
    pub base_url: String,
    pub profile: String,
    pub max_concurrency: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PythonConfig {
    pub algorithm_dir: String,
    pub matching: AlgorithmConfig,
    pub dispatch: AlgorithmConfig,
    pub h3_resolution: u8,
}

/// Algorithm-neutral configuration passed unchanged to Python.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AlgorithmConfig {
    pub name: String,
    #[serde(default)]
    pub options: Map<String, Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct VisualizationConfig {
    pub host: String,
    pub port: u16,
    pub vector_style_url: String,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let mut config: Self = toml::from_str(&text).context("parse config.toml")?;
        config.project_dir = path.canonicalize()?.parent().unwrap().to_path_buf();
        Ok(config)
    }

    pub fn resolve(&self, value: &str) -> PathBuf {
        let path = Path::new(value);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.project_dir.join(path)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_algorithm_neutral_options() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        for name in ["config.toml", "config.smoke.toml"] {
            let config = Config::load(&root.join(name)).unwrap();
            assert_eq!(config.python.matching.name, "maximum");
            assert!(config.python.matching.options.is_empty());
            assert!(!config.python.dispatch.name.is_empty());
        }
    }
}
