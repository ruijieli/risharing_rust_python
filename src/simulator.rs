use std::{
    collections::{HashMap, VecDeque},
    path::Path,
};

use anyhow::{Context, Result};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Serialize;

use crate::{
    config::Config,
    grid::{DispatchTarget, Grid},
    model::{Car, CarState, Passenger, Point, WaitingPassenger},
    osrm::{OsrmClient, interpolate},
};

#[derive(Serialize)]
struct AlgorithmPoint<'a> {
    id: &'a str,
    lat: f64,
    lon: f64,
}

#[derive(Serialize)]
pub struct MatchingInput<'a> {
    cars: Vec<AlgorithmPoint<'a>>,
    passengers: Vec<AlgorithmPoint<'a>>,
    threshold_m: f64,
}

pub struct Simulator {
    pub config: Config,
    pub cars: Vec<Car>,
    pub passengers: Vec<Passenger>,
    pub waiting: Vec<WaitingPassenger>,
    pub current_time: u64,
    pub total_served: u64,
    next_passenger: usize,
    osrm: OsrmClient,
    grid: Grid,
    pending_passengers: Vec<Passenger>,
}

impl Simulator {
    pub fn new(config: Config) -> Result<Self> {
        let passengers = load_passengers(
            &config.resolve(&config.simulation.passenger_file),
            config.simulation.end_time_s,
            config.simulation.sample_size,
            config.simulation.random_seed,
        )?;
        let mut rng = StdRng::seed_from_u64(config.simulation.random_seed);
        let cars = (0..config.simulation.num_cars)
            .map(|index| {
                Car::new(
                    format!("car_{index}"),
                    Point {
                        lat: rng.random_range(config.geo.min_lat..config.geo.max_lat),
                        lon: rng.random_range(config.geo.min_lon..config.geo.max_lon),
                    },
                )
            })
            .collect();
        let osrm = OsrmClient::new(config.osrm.clone())?;
        let grid = Grid::new(&config)?;
        Ok(Self {
            current_time: config.simulation.start_time_s,
            config,
            cars,
            passengers,
            waiting: Vec::new(),
            total_served: 0,
            next_passenger: 0,
            osrm,
            grid,
            pending_passengers: Vec::new(),
        })
    }

    pub async fn validate_osrm(&self) -> Result<()> {
        let Some(car) = self.cars.first().and_then(Car::position) else {
            return Ok(());
        };
        let Some(passenger) = self.passengers.first() else {
            return Ok(());
        };
        let route = self.osrm.route(car, passenger.start()).await;
        anyhow::ensure!(
            !route.points.is_empty(),
            "OSRM did not return a valid Chengdu route"
        );
        Ok(())
    }

    /// Return the matching candidates for the current 10-second batch.
    /// Python owns the algorithm choice and returns only `(car_id, passenger_id)` pairs.
    pub fn matching_input(&mut self) -> MatchingInput<'_> {
        self.pending_passengers = self.current_passengers();
        let cars = self
            .cars
            .iter()
            .filter(|car| matches!(car.state, CarState::Idle | CarState::EmptyTrip))
            .filter_map(|car| {
                car.position().map(|p| AlgorithmPoint {
                    id: &car.id,
                    lat: p.lat,
                    lon: p.lon,
                })
            })
            .collect();
        let passengers = self
            .pending_passengers
            .iter()
            .map(|p| AlgorithmPoint {
                id: &p.id,
                lat: p.start_lat,
                lon: p.start_lon,
            })
            .collect();
        MatchingInput {
            cars,
            passengers,
            threshold_m: self.config.simulation.distance_threshold_m,
        }
    }

    /// Apply matching without advancing time, so the policy can observe the
    /// exact idle fleet that is available for dispatch.
    pub async fn apply_matches(&mut self, matches: Vec<(String, String)>) -> Result<f64> {
        let served_before = self.total_served;
        let passengers = std::mem::take(&mut self.pending_passengers);
        if !passengers.is_empty() {
            self.plan_matches(matches, passengers).await?;
        }
        Ok((self.total_served - served_before) as f64)
    }

    /// Dispatch the current idle fleet, then complete one 10-second micro-step.
    pub async fn advance(&mut self, proportions: Option<&[Vec<f64>]>) -> Result<()> {
        if let Some(proportions) = proportions {
            self.dispatch_proportions(proportions).await?;
        }
        self.update_positions();
        self.current_time += self.config.simulation.batch_interval_s;
        Ok(())
    }

    /// Execute an ordinary matching-only micro-step between dispatch boundaries.
    pub async fn advance_step(&mut self, matches: Vec<(String, String)>) -> Result<f64> {
        let reward = self.apply_matches(matches).await?;
        self.advance(None).await?;
        Ok(reward)
    }

    pub fn observation(&self) -> Vec<f32> {
        self.grid.observation(&self.cars)
    }
    pub fn num_grids(&self) -> usize {
        self.grid.len()
    }
    pub fn grid_cell_ids(&self) -> Vec<String> {
        self.grid.cell_ids()
    }
    pub fn done(&self) -> bool {
        self.current_time >= self.config.simulation.end_time_s
    }

    fn current_passengers(&mut self) -> Vec<Passenger> {
        let minimum = self
            .current_time
            .saturating_sub(self.config.simulation.batch_interval_s) as f64;
        let mut result = Vec::new();
        while self.next_passenger < self.passengers.len() {
            let passenger = &self.passengers[self.next_passenger];
            if passenger.boarding_time > self.current_time as f64 {
                break;
            }
            if passenger.boarding_time >= minimum {
                result.push(passenger.clone());
            }
            self.next_passenger += 1;
        }
        result
    }

    async fn plan_matches(
        &mut self,
        matches: Vec<(String, String)>,
        passengers: Vec<Passenger>,
    ) -> Result<()> {
        let passenger_map: HashMap<String, Passenger> =
            passengers.into_iter().map(|p| (p.id.clone(), p)).collect();
        let car_map: HashMap<String, usize> = self
            .cars
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id.clone(), i))
            .collect();
        let valid: Vec<(usize, Passenger)> = matches
            .into_iter()
            .filter_map(|(car_id, pax_id)| {
                Some((*car_map.get(&car_id)?, passenger_map.get(&pax_id)?.clone()))
            })
            .collect();
        let mut requests = Vec::with_capacity(valid.len() * 2);
        for (car_index, passenger) in &valid {
            requests.push((self.cars[*car_index].position().unwrap(), passenger.start()));
            requests.push((passenger.start(), passenger.end()));
        }
        let routes = self.osrm.routes(requests).await;
        for (match_index, (car_index, passenger)) in valid.into_iter().enumerate() {
            let pickup = &routes[match_index * 2];
            let trip = &routes[match_index * 2 + 1];
            if pickup.points.is_empty() || trip.points.is_empty() {
                continue;
            }
            let pickup_points = interpolate(
                &pickup.points,
                pickup.duration_s,
                self.config.simulation.position_update_interval_s as f64,
            );
            let trip_points = interpolate(
                &trip.points,
                trip.duration_s,
                self.config.simulation.position_update_interval_s as f64,
            );
            let car = &mut self.cars[car_index];
            car.points = VecDeque::from(pickup_points);
            car.next_points = trip_points;
            car.pickup_points_remaining = car.points.len();
            car.state = CarState::EnRouteToPickup;
            car.served_pax_num += 1;
            self.waiting.push(WaitingPassenger {
                passenger,
                pickup_wait_time: car.pickup_points_remaining,
            });
            self.total_served += 1;
        }
        Ok(())
    }

    async fn dispatch_proportions(&mut self, proportions: &[Vec<f64>]) -> Result<()> {
        let idle_indices: Vec<usize> = self
            .cars
            .iter()
            .enumerate()
            .filter(|(_, car)| car.state == CarState::Idle && car.position().is_some())
            .map(|(index, _)| index)
            .collect();
        if idle_indices.is_empty() {
            return Ok(());
        }

        anyhow::ensure!(
            proportions.len() == self.grid.len(),
            "proportion matrix has {} rows; expected {}",
            proportions.len(),
            self.grid.len()
        );
        let mut cars_by_origin = vec![Vec::new(); self.grid.len()];
        for &car_index in &idle_indices {
            let origin = self
                .grid
                .point_index(self.cars[car_index].position().unwrap());
            cars_by_origin[origin].push(car_index);
        }
        let mut targets = Vec::with_capacity(idle_indices.len());
        for origin in 0..self.grid.len() {
            let row = &proportions[origin];
            anyhow::ensure!(
                row.len() == self.grid.len(),
                "proportion matrix row {origin} has {} columns; expected {}",
                row.len(),
                self.grid.len()
            );
            let counts = proportion_row_to_counts(row, cars_by_origin[origin].len(), origin)?;
            let mut cursor = 0;
            for (destination, &count) in counts.iter().enumerate() {
                let end = cursor + count;
                if destination != origin {
                    let point = self.grid.center(destination).expect("validated grid index");
                    targets.extend(
                        cars_by_origin[origin][cursor..end]
                            .iter()
                            .map(|&car_index| DispatchTarget {
                                car_id: self.cars[car_index].id.clone(),
                                point,
                            }),
                    );
                }
                cursor = end;
            }
        }
        self.dispatch_targets(&targets).await
    }

    async fn dispatch_targets(&mut self, targets: &[DispatchTarget]) -> Result<()> {
        let idle_indices: Vec<usize> = self
            .cars
            .iter()
            .enumerate()
            .filter(|(_, car)| car.state == CarState::Idle && car.position().is_some())
            .map(|(index, _)| index)
            .collect();
        let index_by_id: HashMap<String, usize> = idle_indices
            .iter()
            .map(|&i| (self.cars[i].id.clone(), i))
            .collect();
        let valid: Vec<(usize, Point)> = targets
            .iter()
            .filter_map(|target| Some((*index_by_id.get(&target.car_id)?, target.point)))
            .collect();
        let requests = valid
            .iter()
            .map(|(index, target)| (self.cars[*index].position().unwrap(), *target))
            .collect();
        let routes = self.osrm.routes(requests).await;
        for ((car_index, _), route) in valid.into_iter().zip(routes) {
            if route.points.is_empty() {
                continue;
            }
            let points = interpolate(
                &route.points,
                route.duration_s,
                self.config.simulation.position_update_interval_s as f64,
            );
            let car = &mut self.cars[car_index];
            car.points = VecDeque::from(points);
            car.pickup_points_remaining = car.points.len();
            car.state = CarState::EmptyTrip;
        }
        Ok(())
    }

    fn update_positions(&mut self) {
        for car in &mut self.cars {
            match car.state {
                CarState::Idle => car.idle_time += 1,
                CarState::EnRouteToPickup => {
                    car.en_route_to_pickup_time += 1;
                    if car.pickup_points_remaining > 1 {
                        car.points.pop_front();
                        car.pickup_points_remaining -= 1;
                    } else if car.pickup_points_remaining == 1 {
                        car.points.pop_front();
                        car.state = CarState::OnTrip;
                        car.points.extend(car.next_points.drain(..));
                        car.pickup_points_remaining = car.points.len();
                    }
                }
                CarState::OnTrip => {
                    car.on_trip_time += 1;
                    if car.pickup_points_remaining > 1 {
                        car.points.pop_front();
                        car.pickup_points_remaining -= 1;
                    } else if car.pickup_points_remaining == 1 {
                        car.state = CarState::Idle;
                    }
                }
                CarState::EmptyTrip => {
                    car.empty_trip_time += 1;
                    if car.pickup_points_remaining > 1 {
                        car.points.pop_front();
                        car.pickup_points_remaining -= 1;
                    } else if car.pickup_points_remaining == 1 {
                        car.state = CarState::Idle;
                    }
                }
            }
        }
        // Deliberately mirrors the current Python list-removal behavior for
        // comparison. Adjacent entries can be skipped after a removal.
        let mut index = 0;
        while index < self.waiting.len() {
            if self.waiting[index].pickup_wait_time > 0 {
                self.waiting[index].pickup_wait_time -= 1;
            } else {
                self.waiting.remove(index);
            }
            index += 1;
        }
    }

    pub fn save_results(&self, path: &Path) -> Result<()> {
        let mut writer = csv::Writer::from_path(path)?;
        writer.write_record([
            "id",
            "served_pax_num",
            "idle_time",
            "en_route_to_pickup_time",
            "on_trip_time",
            "empty_trip_time",
        ])?;
        for car in &self.cars {
            writer.serialize((
                &car.id,
                car.served_pax_num,
                car.idle_time,
                car.en_route_to_pickup_time,
                car.on_trip_time,
                car.empty_trip_time,
            ))?;
        }
        writer.flush()?;
        Ok(())
    }
}

/// Convert one policy row to integer vehicle counts exactly once, using the
/// idle fleet measured after matching at the dispatch boundary.
fn proportion_row_to_counts(row: &[f64], available: usize, origin: usize) -> Result<Vec<usize>> {
    let mut result = vec![0; row.len()];
    anyhow::ensure!(
        row.iter().all(|value| value.is_finite() && *value >= 0.0),
        "proportion matrix contains invalid values"
    );
    if available == 0 {
        return Ok(result);
    }
    let total: f64 = row.iter().sum();
    if total <= f64::EPSILON {
        result[origin] = available;
        return Ok(result);
    }
    let quotas: Vec<f64> = row
        .iter()
        .map(|&proportion| available as f64 * proportion / total)
        .collect();
    for (destination, quota) in quotas.iter().enumerate() {
        result[destination] = quota.floor() as usize;
    }
    let assigned: usize = result.iter().sum();
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| {
        let fraction_a = quotas[a] - quotas[a].floor();
        let fraction_b = quotas[b] - quotas[b].floor();
        fraction_b.total_cmp(&fraction_a).then_with(|| a.cmp(&b))
    });
    for &destination in order.iter().take(available - assigned) {
        result[destination] += 1;
    }
    Ok(result)
}

fn load_passengers(
    path: &Path,
    end_time: u64,
    sample_size: Option<usize>,
    seed: u64,
) -> Result<Vec<Passenger>> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("open passenger CSV {}", path.display()))?;
    let mut passengers: Vec<Passenger> = reader
        .deserialize::<Passenger>()
        .filter_map(|row| match row {
            Ok(p) if p.boarding_time < end_time as f64 => Some(Ok(p)),
            Ok(_) => None,
            Err(e) => Some(Err(e)),
        })
        .collect::<std::result::Result<_, _>>()?;
    if let Some(size) = sample_size.filter(|&size| size < passengers.len()) {
        passengers.shuffle(&mut StdRng::seed_from_u64(seed));
        passengers.truncate(size);
    }
    passengers.sort_by(|a, b| a.boarding_time.total_cmp(&b.boarding_time));
    Ok(passengers)
}

#[cfg(test)]
mod tests {
    use super::proportion_row_to_counts;

    #[test]
    fn converts_proportions_using_post_matching_idle_vehicles() {
        let row = proportion_row_to_counts(&[0.7, 0.2, 0.1], 6, 0).unwrap();
        assert_eq!(row.iter().sum::<usize>(), 6);
        assert_eq!(row, vec![4, 1, 1]);
    }
}
