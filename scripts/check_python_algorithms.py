#!/usr/bin/env python3
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "python_algorithms"))

import dispatch
import matching


@matching.register("first_pair")
def first_pair(cars, passengers, *, threshold_m, options, project_dir):
    assert threshold_m == 123.0 and options == {"student_option": 7}
    assert project_dir == str(root)
    return [[cars[0]["id"], passengers[0]["id"]]]


@dispatch.register("stay")
def stay(state, options, project_dir):
    assert options == {"student_option": 9}
    assert project_dir == str(root)
    return {"kind": "proportions", "values": __import__("numpy").eye(state.size).tolist()}

matching_input = {
    "cars": [{"id": "car_0", "lat": 30.66, "lon": 104.06}],
    "passengers": [{"id": "pax_0", "lat": 30.661, "lon": 104.061}],
}
matching_input["threshold_m"] = 1000.0
matching_input["options"] = {}
matches = matching.match_vehicles(
    matching_input["cars"], matching_input["passengers"], matching_input["threshold_m"],
    "maximum", matching_input["options"], str(root),
)
assert matches == [["car_0", "pax_0"]], matches
nearest_cars = [
    {"id": "car_0", "lat": 30.6600, "lon": 104.0600},
    {"id": "car_1", "lat": 30.6700, "lon": 104.0700},
]
nearest_passengers = [
    {"id": "pax_0", "lat": 30.6600, "lon": 104.0600},
    {"id": "pax_1", "lat": 30.6701, "lon": 104.0701},
]
nearest_matches = matching.match_vehicles(
    nearest_cars, nearest_passengers, 1000.0, "nearest",
    {"strategy": "global", "max_matches": 1}, str(root),
)
assert nearest_matches == [["car_0", "pax_0"]], nearest_matches
assert matching.available_algorithms() == ("first_pair", "maximum", "nearest")
custom_matching_input = dict(matching_input, threshold_m=123.0,
                             options={"student_option": 7}, project_dir=str(root))
custom_matches = matching.match_vehicles(
    custom_matching_input["cars"], custom_matching_input["passengers"],
    custom_matching_input["threshold_m"], "first_pair", custom_matching_input["options"],
    custom_matching_input["project_dir"],
)
assert custom_matches == [["car_0", "pax_0"]], custom_matches

dispatch_input = {
    "observation": [1.0, 0.0, 0.0],
    "options": {},
    "project_dir": str(root),
}
action = dispatch.choose_action(
    dispatch_input["observation"], "random", dispatch_input["options"], dispatch_input["project_dir"]
)
assert action["kind"] == "proportions" and len(action["values"]) == 3, action
assert dispatch.available_algorithms() == ("none", "ppo", "random", "sac", "stay")
custom_dispatch_input = dict(dispatch_input, options={"student_option": 9})
custom_action = dispatch.choose_action(
    custom_dispatch_input["observation"], "stay", custom_dispatch_input["options"],
    custom_dispatch_input["project_dir"],
)
assert custom_action["values"] == __import__("numpy").eye(3).tolist(), custom_action
none_action = dispatch.evaluation_action(
    __import__("numpy").asarray(dispatch_input["observation"], dtype="float32"),
    "none",
    dispatch_input["options"],
    dispatch_input["project_dir"],
)
assert none_action.tolist() == __import__("numpy").eye(3).tolist(), none_action
print("Python algorithm interface: OK")
