use std::{path::Path, sync::Arc};

use anyhow::Result;
use tokio::sync::RwLock;

use _rust_core::{
    config::Config,
    grid::Grid,
    simulator::Simulator,
    web::{self, SharedSnapshot},
};

#[tokio::main]
async fn main() -> Result<()> {
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config.toml".to_string());
    let config = Config::load(Path::new(&config_path))?;
    let snapshot: SharedSnapshot = Arc::new(RwLock::new(None));

    let server = if config.simulation.visualization {
        let web_config = config.visualization.clone();
        let web_snapshot = snapshot.clone();
        let grid = Grid::new(&config)?.geojson();
        Some(tokio::spawn(async move {
            web::serve(web_config, web_snapshot, grid).await
        }))
    } else {
        None
    };

    let mut simulator = Simulator::new(config.clone())?;
    simulator
        .run(config.simulation.visualization.then_some(snapshot))
        .await?;

    let output_dir = config.project_dir.join("output");
    std::fs::create_dir_all(&output_dir)?;
    let output = output_dir.join(format!(
        "{}demand_{}cars_0iter.csv",
        simulator.passengers.len(),
        simulator.cars.len()
    ));
    simulator.save_results(&output)?;
    println!("served_paxs = {}", simulator.total_served);
    println!("total_paxs = {}", simulator.passengers.len());
    println!("result = {}", output.display());

    if config.simulation.keep_visualization_alive {
        if let Some(server) = server {
            server.await??;
        }
    } else if let Some(server) = server {
        server.abort();
    }
    Ok(())
}
