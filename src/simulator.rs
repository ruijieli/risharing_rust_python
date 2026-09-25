use std::{
    collections::{HashMap, VecDeque},
    path::Path,
};

use anyhow::{Context, Result};
use rand::{
    Rng, SeedableRng,
    distr::{Distribution, weighted::WeightedIndex},
    rngs::StdRng,
    seq::SliceRandom,
};

use crate::{
    config::Config,
    grid::{DispatchAction, DispatchTarget, Grid},
    model::{Car, CarState, Passenger, Point, WaitingPassenger},
    osrm::{OsrmClient, interpolate},
    python_bridge::PythonAlgorithms,
    web::{SharedSnapshot, build_snapshot},
};

pub struct Simulator {
    pub config: Config,
    pub cars: Vec<Car>,
    pub passengers: Vec<Passenger>,
    pub waiting: Vec<WaitingPassenger>,
    pub current_time: u64,
    pub total_served: u64,
    next_passenger: usize,
    osrm: OsrmClient,
    algorithms: PythonAlgorithms,
    grid: Grid,
    rng: StdRng,
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
        let algorithms = PythonAlgorithms::new(&config);
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
            algorithms,
            grid,
            rng,
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

    pub async fn run(&mut self, snapshot: Option<SharedSnapshot>) -> Result<()> {
        self.validate_osrm().await?;
        while self.current_time < self.config.simulation.end_time_s {
            self.step().await?;
            if let Some(shared) = &snapshot {
                *shared.write().await = Some(build_snapshot(&self.waiting, &self.cars));
                let delay = self.config.simulation.visualization_step_delay_ms;
                if delay > 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                }
            }
        }
        Ok(())
    }

    async fn step(&mut self) -> Result<()> {
        let passengers = self.current_passengers();
        let available_indices: Vec<usize> = self
            .cars
            .iter()
            .enumerate()
            .filter(|(_, car)| {
                matches!(car.state, CarState::Idle | CarState::EmptyTrip)
                    && car.position().is_some()
            })
            .map(|(index, _)| index)
            .collect();
        if !available_indices.is_empty() && !passengers.is_empty() {
            let car_refs: Vec<&Car> = available_indices
                .iter()
                .map(|&index| &self.cars[index])
                .collect();
            let matches = self.algorithms.matching(
                &car_refs,
                &passengers,
                self.config.simulation.distance_threshold_m,
            )?;
            self.plan_matches(matches, passengers).await?;
        }

        if (self.current_time - self.config.simulation.start_time_s)
            .is_multiple_of(self.config.simulation.dispatch_cycle_s)
        {
            self.dispatch().await?;
        }
        self.update_positions();
        self.current_time += self.config.simulation.batch_interval_s;
        Ok(())
    }

    /// Advance one RL decision period. Python chooses dispatch targets once;
    /// matching, routing, time and vehicle transitions remain in Rust.
    pub async fn training_cycle(&mut self, action: DispatchAction) -> Result<f64> {
        let served_before = self.total_served;
        let micro_steps = (self.config.simulation.dispatch_cycle_s
            / self.config.simulation.batch_interval_s)
            .max(1);
        for micro_step in 0..micro_steps {
            if self.current_time >= self.config.simulation.end_time_s {
                break;
            }
            self.match_current_batch().await?;
            if micro_step == 0 {
                self.dispatch_action(&action).await?;
            }
            self.update_positions();
            self.current_time += self.config.simulation.batch_interval_s;
        }
        Ok((self.total_served - served_before) as f64)
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

    async fn match_current_batch(&mut self) -> Result<()> {
        let passengers = self.current_passengers();
        let available_indices: Vec<usize> = self
            .cars
            .iter()
            .enumerate()
            .filter(|(_, car)| {
                matches!(car.state, CarState::Idle | CarState::EmptyTrip)
                    && car.position().is_some()
            })
            .map(|(index, _)| index)
            .collect();
        if !available_indices.is_empty() && !passengers.is_empty() {
            let car_refs: Vec<&Car> = available_indices
                .iter()
                .map(|&index| &self.cars[index])
                .collect();
            let matches = self.algorithms.matching(
                &car_refs,
                &passengers,
                self.config.simulation.distance_threshold_m,
            )?;
            self.plan_matches(matches, passengers).await?;
        }
        Ok(())
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

    async fn dispatch(&mut self) -> Result<()> {
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
        let action = self.algorithms.dispatch(&self.observation())?;
        self.dispatch_action(&action).await
    }

    async fn dispatch_action(&mut self, action: &DispatchAction) -> Result<()> {
        let idle_indices: Vec<usize> = self
            .cars
            .iter()
            .enumerate()
            .filter(|(_, car)| car.state == CarState::Idle && car.position().is_some())
            .map(|(index, _)| index)
            .collect();
        if idle_indices.is_empty() || matches!(action, DispatchAction::None) {
            return Ok(());
        }

        let targets = match action {
            DispatchAction::None => Vec::new(),
            DispatchAction::Discrete { value } => {
                if *value == 0 {
                    Vec::new()
                } else {
                    let destination = value - 1;
                    let point = self.grid.center(destination).with_context(|| {
                        format!("discrete dispatch action {value} exceeds action space")
                    })?;
                    idle_indices
                        .iter()
                        .map(|&i| DispatchTarget {
                            car_id: self.cars[i].id.clone(),
                            point,
                        })
                        .collect()
                }
            }
            DispatchAction::Matrix { values } => {
                anyhow::ensure!(
                    values.len() == self.grid.len(),
                    "dispatch matrix has {} rows; expected {}",
                    values.len(),
                    self.grid.len()
                );
                let mut targets = Vec::with_capacity(idle_indices.len());
                for &car_index in &idle_indices {
                    let origin = self
                        .grid
                        .point_index(self.cars[car_index].position().unwrap());
                    let row = &values[origin];
                    anyhow::ensure!(
                        row.len() == self.grid.len(),
                        "dispatch matrix row {origin} has {} columns; expected {}",
                        row.len(),
                        self.grid.len()
                    );
                    anyhow::ensure!(
                        row.iter().all(|v| v.is_finite() && *v >= 0.0),
                        "dispatch matrix contains invalid values"
                    );
                    let Ok(distribution) = WeightedIndex::new(row) else {
                        continue;
                    };
                    let destination = distribution.sample(&mut self.rng);
                    targets.push(DispatchTarget {
                        car_id: self.cars[car_index].id.clone(),
                        point: self.grid.center(destination).expect("validated grid index"),
                    });
                }
                targets
            }
        };
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
