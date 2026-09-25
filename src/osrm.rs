use std::sync::Arc;

use futures::{StreamExt, stream};
use serde::Deserialize;

use crate::{
    config::OsrmConfig,
    model::{Point, RouteResult},
};

#[derive(Clone)]
pub struct OsrmClient {
    client: reqwest::Client,
    config: OsrmConfig,
}

#[derive(Deserialize)]
struct Response {
    routes: Vec<Route>,
}
#[derive(Deserialize)]
struct Route {
    distance: f64,
    duration: f64,
    geometry: Geometry,
}
#[derive(Deserialize)]
struct Geometry {
    coordinates: Vec<[f64; 2]>,
}

impl OsrmClient {
    pub fn new(config: OsrmConfig) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            // OSRM is a local service. Explicitly bypass macOS/Conda proxy
            // settings; otherwise localhost requests can be sent to an HTTP
            // proxy and return 502 without ever reaching the OSRM container.
            .no_proxy()
            .pool_max_idle_per_host(config.max_concurrency)
            .timeout(std::time::Duration::from_secs(15))
            .build()?;
        Ok(Self { client, config })
    }

    pub async fn route(&self, start: Point, end: Point) -> RouteResult {
        let url = format!(
            "{}/route/v1/{}/{},{};{},{}",
            self.config.base_url.trim_end_matches('/'),
            self.config.profile,
            start.lon,
            start.lat,
            end.lon,
            end.lat
        );
        let response = match self
            .client
            .get(url)
            .query(&[
                ("overview", "full"),
                ("geometries", "geojson"),
                ("steps", "false"),
            ])
            .send()
            .await
            .and_then(|r| r.error_for_status())
        {
            Ok(response) => response,
            Err(error) => {
                eprintln!("OSRM request failed: {error}");
                return RouteResult::default();
            }
        };
        let data: Response = match response.json().await {
            Ok(data) => data,
            Err(error) => {
                eprintln!("OSRM response decode failed: {error}");
                return RouteResult::default();
            }
        };
        let Some(route) = data.routes.into_iter().next() else {
            return RouteResult::default();
        };
        if route.distance <= 0.0 || route.duration <= 0.0 {
            return RouteResult::default();
        }
        RouteResult {
            distance_m: route.distance,
            duration_s: route.duration,
            points: route
                .geometry
                .coordinates
                .into_iter()
                .map(|c| Point {
                    lon: c[0],
                    lat: c[1],
                })
                .collect(),
        }
    }

    pub async fn routes(&self, requests: Vec<(Point, Point)>) -> Vec<RouteResult> {
        let client = Arc::new(self.clone());
        let concurrency = self.config.max_concurrency.max(1);
        stream::iter(requests.into_iter().enumerate())
            .map(|(index, (start, end))| {
                let client = client.clone();
                async move { (index, client.route(start, end).await) }
            })
            .buffer_unordered(concurrency)
            .fold(Vec::new(), |mut values, (index, route)| async move {
                values.push((index, route));
                values
            })
            .await
            .pipe(|mut values| {
                values.sort_by_key(|(index, _)| *index);
                values.into_iter().map(|(_, route)| route).collect()
            })
    }
}

trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(Self) -> T) -> T {
        f(self)
    }
}
impl<T> Pipe for T {}

pub fn interpolate(points: &[Point], total_time: f64, interval: f64) -> Vec<Point> {
    if points.is_empty() || total_time <= 0.0 {
        return points.to_vec();
    }
    let target = ((total_time / interval).floor() as usize + 1).max(2);
    if points.len() > target {
        let step = (points.len() - 1) as f64 / (target - 1) as f64;
        return (0..target)
            .map(|i| points[(i as f64 * step).round() as usize])
            .collect();
    }
    (0..target)
        .map(|i| {
            let position = i as f64 * (points.len() - 1) as f64 / (target - 1) as f64;
            let left = position.floor() as usize;
            let right = position.ceil() as usize;
            let ratio = position - left as f64;
            Point {
                lat: points[left].lat + (points[right].lat - points[left].lat) * ratio,
                lon: points[left].lon + (points[right].lon - points[left].lon) * ratio,
            }
        })
        .collect()
}
