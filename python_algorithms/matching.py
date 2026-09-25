"""User-editable vehicle/passenger matching algorithms."""

import networkx as nx
import numpy as np


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


def match_vehicles(cars: list[dict], passengers: list[dict],
                   threshold_m: float, algorithm: str = "maximum") -> list[list[str]]:
    """Return ``[[car_id, passenger_id], ...]`` without changing simulator state."""
    if not cars or not passengers:
        return []
    if algorithm != "maximum":
        raise ValueError(f"Unknown matching algorithm: {algorithm}")

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
