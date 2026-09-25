use std::{net::SocketAddr, sync::Arc};

use axum::{Json, Router, extract::State, response::Html, routing::get};
use serde::Serialize;
use tokio::sync::RwLock;

use crate::{
    config::VisualizationConfig,
    model::{Car, CarState, WaitingPassenger},
};

#[derive(Clone)]
struct AppState {
    style_url: String,
    snapshot: SharedSnapshot,
    grid: serde_json::Value,
}

#[derive(Clone, Default, Serialize)]
pub struct Snapshot {
    pub pax: Vec<NamedValue>,
    pub car_on_trip: Vec<NamedValue>,
    pub car_idle: Vec<NamedValue>,
    pub car_en_route_to_pickup: Vec<NamedValue>,
    pub car_empty_trip: Vec<NamedValue>,
    #[serde(rename = "geoCoordMap")]
    pub geo_coord_map: std::collections::BTreeMap<String, [f64; 2]>,
}

#[derive(Clone, Serialize)]
pub struct NamedValue {
    pub name: String,
    pub value: u8,
}

pub type SharedSnapshot = Arc<RwLock<Option<Snapshot>>>;

pub fn build_snapshot(waiting: &[WaitingPassenger], cars: &[Car]) -> Snapshot {
    let mut snapshot = Snapshot::default();
    for item in waiting {
        let name = format!("乘客{}", item.passenger.id);
        snapshot.pax.push(NamedValue {
            name: name.clone(),
            value: 1,
        });
        snapshot
            .geo_coord_map
            .insert(name, [item.passenger.start_lon, item.passenger.start_lat]);
    }
    for car in cars {
        let Some(position) = car.position() else {
            continue;
        };
        let name = format!("司机{}", car.id);
        let item = NamedValue {
            name: name.clone(),
            value: 1,
        };
        match car.state {
            CarState::Idle => snapshot.car_idle.push(item),
            CarState::EnRouteToPickup => snapshot.car_en_route_to_pickup.push(item),
            CarState::OnTrip => snapshot.car_on_trip.push(item),
            CarState::EmptyTrip => snapshot.car_empty_trip.push(item),
        }
        snapshot
            .geo_coord_map
            .insert(name, [position.lon, position.lat]);
    }
    snapshot
}

async fn index(State(state): State<AppState>) -> Html<String> {
    Html(include_str!("../web/index.html").replace("__VECTOR_STYLE_URL__", &state.style_url))
}

async fn coordinates(State(state): State<AppState>) -> Json<serde_json::Value> {
    match state.snapshot.read().await.clone() {
        Some(value) => Json(serde_json::to_value(value).unwrap()),
        None => Json(serde_json::json!({"status":"init", "message":"系统初始化中，请稍候..."})),
    }
}

async fn grid_geojson(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(state.grid.clone())
}

pub async fn serve(
    config: VisualizationConfig,
    snapshot: SharedSnapshot,
    grid: serde_json::Value,
) -> anyhow::Result<()> {
    let address: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    serve_with_listener(config, snapshot, grid, listener).await
}

pub async fn serve_with_listener(
    config: VisualizationConfig,
    snapshot: SharedSnapshot,
    grid: serde_json::Value,
    listener: tokio::net::TcpListener,
) -> anyhow::Result<()> {
    let state = AppState {
        style_url: config.vector_style_url.clone(),
        snapshot,
        grid,
    };
    let app = Router::new()
        .route("/", get(index))
        .route("/grid", get(grid_geojson))
        .route("/get_coordinates", get(coordinates))
        .with_state(state);
    let address = listener.local_addr()?;
    println!("Visualization: http://{address}");
    axum::serve(listener, app).await?;
    Ok(())
}
