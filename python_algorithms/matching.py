"""User-editable vehicle/passenger matching algorithm registry."""

from __future__ import annotations

from typing import Callable

import networkx as nx
import numpy as np

MatchingAlgorithm = Callable[..., list]
_ALGORITHMS: dict[str, MatchingAlgorithm] = {}


def register(name: str):
    """Register a matching algorithm without changing the Rust extension."""
    def decorator(function: MatchingAlgorithm) -> MatchingAlgorithm:
        if name in _ALGORITHMS:
            raise ValueError(f"Matching algorithm already registered: {name}")
        _ALGORITHMS[name] = function
        return function
    return decorator


def available_algorithms() -> tuple[str, ...]:
    return tuple(sorted(_ALGORITHMS))


def _haversine(car_points: np.ndarray, pax_points: np.ndarray) -> np.ndarray:
    radius = 6_371_000.0
    car_lat = np.radians(car_points[:, 0])[:, None]
    car_lon = np.radians(car_points[:, 1])[:, None]
    pax_lat = np.radians(pax_points[:, 0])
    pax_lon = np.radians(pax_points[:, 1])
    d_lat = pax_lat - car_lat
    d_lon = pax_lon - car_lon
    value = np.sin(d_lat / 2) ** 2 + np.cos(car_lat) * np.cos(pax_lat) * np.sin(d_lon / 2) ** 2
    return radius * 2 * np.arctan2(np.sqrt(value), np.sqrt(1 - value))


@register("maximum")
def maximum_matching(cars: list[dict], passengers: list[dict], *,
                     threshold_m: float, options: dict,
                     project_dir: str) -> list[list[str]]:
    """Maximum-cardinality bipartite matching within ``threshold_m``."""
    del options, project_dir
    car_points = np.asarray([[item["lat"], item["lon"]] for item in cars], dtype=np.float64)
    pax_points = np.asarray([[item["lat"], item["lon"]] for item in passengers], dtype=np.float64)
    distances = _haversine(car_points, pax_points)
    valid = np.where((distances < threshold_m) & (distances > 0))
    graph = nx.Graph()
    car_ids = [item["id"] for item in cars]
    pax_ids = [item["id"] for item in passengers]
    graph.add_nodes_from(car_ids, bipartite=0)
    graph.add_nodes_from(pax_ids, bipartite=1)
    for car_index, pax_index in zip(*valid):
        graph.add_edge(car_ids[car_index], pax_ids[pax_index], weight=float(distances[car_index, pax_index]))
    car_id_set = set(car_ids)
    result = nx.algorithms.bipartite.maximum_matching(graph, top_nodes=car_id_set)
    return [[car_id, pax_id] for car_id, pax_id in result.items() if car_id in car_id_set]


@register("nearest")
def nearest_matching(cars: list[dict], passengers: list[dict], *,
                     threshold_m: float, options: dict,
                     project_dir: str) -> list[list[str]]:
    """Greedily match nearby pairs within ``threshold_m``.

    ``strategy="passenger"`` processes passengers in input order and assigns
    each one its nearest unused car. ``strategy="global"`` processes all
    eligible car/passenger pairs from shortest to longest distance.
    """
    del project_dir
    strategy = str(options.get("strategy", "passenger"))
    if strategy not in {"passenger", "global"}:
        raise ValueError("nearest matching option 'strategy' must be passenger or global")
    max_matches = int(options.get("max_matches", 0))
    if max_matches < 0:
        raise ValueError("nearest matching option 'max_matches' must be >= 0")

    car_points = np.asarray([[item["lat"], item["lon"]] for item in cars], dtype=np.float64)
    pax_points = np.asarray(
        [[item["lat"], item["lon"]] for item in passengers], dtype=np.float64
    )
    distances = _haversine(car_points, pax_points)
    eligible = np.isfinite(distances) & (distances <= threshold_m)
    pairs: list[tuple[int, int]] = []

    if strategy == "passenger":
        unused_cars = np.ones(len(cars), dtype=bool)
        for pax_index in range(len(passengers)):
            candidates = np.flatnonzero(eligible[:, pax_index] & unused_cars)
            if candidates.size == 0:
                continue
            candidate_distances = distances[candidates, pax_index]
            car_index = int(candidates[np.argmin(candidate_distances)])
            pairs.append((car_index, pax_index))
            unused_cars[car_index] = False
            if max_matches and len(pairs) >= max_matches:
                break
    else:
        car_indices, pax_indices = np.where(eligible)
        candidates = sorted(
            zip(car_indices.tolist(), pax_indices.tolist()),
            key=lambda pair: (distances[pair], pair[0], pair[1]),
        )
        used_cars: set[int] = set()
        used_passengers: set[int] = set()
        for car_index, pax_index in candidates:
            if car_index in used_cars or pax_index in used_passengers:
                continue
            pairs.append((car_index, pax_index))
            used_cars.add(car_index)
            used_passengers.add(pax_index)
            if max_matches and len(pairs) >= max_matches:
                break

    return [[cars[car]["id"], passengers[pax]["id"]] for car, pax in pairs]


def match_vehicles(cars: list[dict], passengers: list[dict], threshold_m: float,
                   algorithm: str = "maximum", options: dict | None = None,
                   project_dir: str = ".") -> list[list[str]]:
    """Run a registered algorithm and return ``[[car_id, passenger_id], ...]``."""
    if not cars or not passengers:
        return []
    try:
        function = _ALGORITHMS[algorithm]
    except KeyError as error:
        available = ", ".join(available_algorithms())
        raise ValueError(
            f"Unknown matching algorithm: {algorithm}; available: {available}"
        ) from error
    return function(
        cars, passengers, threshold_m=threshold_m, options=options or {},
        project_dir=project_dir,
    )
