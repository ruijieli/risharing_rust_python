use std::collections::HashMap;

use anyhow::{Result, ensure};
use h3o::{CellIndex, LatLng, Resolution};

use crate::{
    config::Config,
    model::{Car, CarState, Point},
};

#[derive(Clone, Debug)]
pub struct Grid {
    resolution: Resolution,
    cells: Vec<CellIndex>,
    index: HashMap<CellIndex, usize>,
    centers: Vec<Point>,
}

#[derive(Clone, Debug)]
pub struct DispatchTarget {
    pub car_id: String,
    pub point: Point,
}

impl Grid {
    pub fn new(config: &Config) -> Result<Self> {
        let resolution = Resolution::try_from(config.python.h3_resolution)
            .map_err(|error| anyhow::anyhow!("invalid H3 resolution: {error:?}"))?;
        let g = &config.geo;
        let center_lat = (g.min_lat + g.max_lat) / 2.0;
        let center_lon = (g.min_lon + g.max_lon) / 2.0;
        let origin = LatLng::new(center_lat, center_lon)
            .map_err(|error| anyhow::anyhow!("invalid grid center: {error:?}"))?
            .to_cell(resolution);
        // Cover a disk larger than the rectangle, then apply the same centroid
        // containment rule as Python h3.geo_to_cells.
        let lat_km = (g.max_lat - g.min_lat).abs() * 111.32 / 2.0;
        let lon_km = (g.max_lon - g.min_lon).abs() * 111.32 * center_lat.to_radians().cos() / 2.0;
        let corner_km = lat_km.hypot(lon_km);
        let radius = (corner_km / resolution.edge_length_km()).ceil() as u32 + 3;
        let candidates: Vec<CellIndex> = origin.grid_disk(radius);
        let mut cells: Vec<_> = candidates
            .into_iter()
            .filter(|cell| {
                let center = LatLng::from(*cell);
                center.lat() >= g.min_lat
                    && center.lat() <= g.max_lat
                    && center.lng() >= g.min_lon
                    && center.lng() <= g.max_lon
            })
            .collect();
        // Keep the exact stable ordering used by Python's sorted H3 strings.
        cells.sort_by_key(ToString::to_string);
        ensure!(!cells.is_empty(), "H3 grid is empty");
        let index = cells
            .iter()
            .enumerate()
            .map(|(i, cell)| (*cell, i))
            .collect();
        let centers = cells
            .iter()
            .map(|cell| {
                let center = LatLng::from(*cell);
                Point {
                    lat: center.lat(),
                    lon: center.lng(),
                }
            })
            .collect();
        Ok(Self {
            resolution,
            cells,
            index,
            centers,
        })
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
    pub fn cell_ids(&self) -> Vec<String> {
        self.cells.iter().map(ToString::to_string).collect()
    }
    pub fn center(&self, index: usize) -> Option<Point> {
        self.centers.get(index).copied()
    }

    pub fn point_index(&self, point: Point) -> usize {
        if let Ok(latlng) = LatLng::new(point.lat, point.lon) {
            let cell = latlng.to_cell(self.resolution);
            if let Some(index) = self.index.get(&cell) {
                return *index;
            }
        }
        let scale = point.lat.to_radians().cos();
        self.centers
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                squared_distance(**a, point, scale).total_cmp(&squared_distance(**b, point, scale))
            })
            .map_or(0, |(index, _)| index)
    }

    pub fn observation(&self, cars: &[Car]) -> Vec<f32> {
        let mut values = vec![0.0; self.len()];
        for car in cars.iter().filter(|car| car.state == CarState::Idle) {
            if let Some(point) = car.position() {
                values[self.point_index(point)] += 1.0;
            }
        }
        values
    }

    // 将所有 H3 六边形网格转换成 GeoJSON 格式，供 Web 前端（如 Leaflet、Mapbox）绘制地图使用。
    pub fn geojson(&self) -> serde_json::Value {
        let features: Vec<_> = self
            .cells
            .iter()
            .enumerate()
            .map(|(index, cell)| {
                let boundary = cell.boundary();
                let mut coordinates: Vec<_> = boundary
                    .iter()
                    .map(|p| serde_json::json!([p.lng(), p.lat()]))
                    .collect();
                if let Some(first) = coordinates.first().cloned() {
                    coordinates.push(first);
                }
                serde_json::json!({
                    "type": "Feature", "id": index,
                    "properties": {"grid_index": index, "cell_id": cell.to_string()},
                    "geometry": {"type": "Polygon", "coordinates": [coordinates]}
                })
            })
            .collect();
        serde_json::json!({"type": "FeatureCollection", "features": features})
    }
}

fn squared_distance(a: Point, b: Point, longitude_scale: f64) -> f64 {
    (a.lat - b.lat).powi(2) + ((a.lon - b.lon) * longitude_scale).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn grid_matches_legacy_python_order() {
        let config = Config::load(Path::new("config.toml")).unwrap();
        let grid = Grid::new(&config).unwrap();
        let ids = grid.cell_ids();
        assert_eq!(ids.len(), 49);
        assert_eq!(ids.first().unwrap(), "8740e31b0ffffff");
        assert_eq!(ids.last().unwrap(), "8740e3cf5ffffff");
    }
}
