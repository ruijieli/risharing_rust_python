use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Point {
    pub lat: f64,
    pub lon: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CarState {
    Idle,
    EnRouteToPickup,
    OnTrip,
    EmptyTrip,
}

#[derive(Debug)]
pub struct Car {
    pub id: String,
    pub points: VecDeque<Point>,
    pub next_points: Vec<Point>,
    pub pickup_points_remaining: usize,
    pub state: CarState,
    pub served_pax_num: u64,
    pub idle_time: u64,
    pub en_route_to_pickup_time: u64,
    pub on_trip_time: u64,
    pub empty_trip_time: u64,
}

impl Car {
    pub fn new(id: String, point: Point) -> Self {
        Self {
            id,
            points: VecDeque::from([point]),
            next_points: Vec::new(),
            pickup_points_remaining: 0,
            state: CarState::Idle,
            served_pax_num: 0,
            idle_time: 0,
            en_route_to_pickup_time: 0,
            on_trip_time: 0,
            empty_trip_time: 0,
        }
    }

    pub fn position(&self) -> Option<Point> {
        self.points.front().copied()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Passenger {
    #[serde(rename = "PassengerID")]
    pub id: String,
    #[serde(rename = "PassengerCount")]
    pub passenger_count: u32,
    #[serde(rename = "BoardingTime")]
    pub boarding_time: f64,
    #[serde(rename = "StartLatitude")]
    pub start_lat: f64,
    #[serde(rename = "StartLongitude")]
    pub start_lon: f64,
    #[serde(rename = "EndLatitude")]
    pub end_lat: f64,
    #[serde(rename = "EndLongitude")]
    pub end_lon: f64,
}

impl Passenger {
    pub fn start(&self) -> Point {
        Point {
            lat: self.start_lat,
            lon: self.start_lon,
        }
    }
    pub fn end(&self) -> Point {
        Point {
            lat: self.end_lat,
            lon: self.end_lon,
        }
    }
}

#[derive(Clone, Debug)]
pub struct WaitingPassenger {
    pub passenger: Passenger,
    pub pickup_wait_time: usize,
}

#[derive(Clone, Debug, Default)]
pub struct RouteResult {
    pub distance_m: f64,
    pub duration_s: f64,
    pub points: Vec<Point>,
}
