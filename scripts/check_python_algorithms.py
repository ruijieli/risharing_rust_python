#!/usr/bin/env python3
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "python_algorithms"))

import api
import dispatch
import matching


@matching.register("first_pair")
def first_pair(cars, passengers, *, threshold_m, options, project_dir):
    assert threshold_m == 123.0 and options == {"student_option": 7}
    assert project_dir == str(root)
    return [[cars[0]["id"], passengers[0]["id"]]]


@dispatch.register("stay", action_mode="continuous")
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
matches = json.loads(api.match_json(json.dumps(matching_input), "maximum"))
assert matches == [["car_0", "pax_0"]], matches
custom_matching_input = dict(matching_input, threshold_m=123.0,
                             options={"student_option": 7}, project_dir=str(root))
custom_matches = json.loads(
    api.match_json(json.dumps(custom_matching_input), "first_pair")
)
assert custom_matches == [["car_0", "pax_0"]], custom_matches

dispatch_input = {
    "observation": [1.0, 0.0, 0.0],
    "options": {},
    "project_dir": str(root),
}
action = json.loads(api.dispatch_action_json(json.dumps(dispatch_input), "random"))
assert action["kind"] == "flow_matrix" and len(action["counts"]) == 3, action
assert [sum(row) for row in action["counts"]] == [1, 0, 0], action
assert dispatch.action_mode("none") == "none"
assert dispatch.action_mode("ppo") == "continuous"
assert dispatch.action_mode("dqn") == "discrete"
custom_dispatch_input = dict(dispatch_input, options={"student_option": 9})
custom_action = json.loads(
    api.dispatch_action_json(json.dumps(custom_dispatch_input), "stay")
)
assert custom_action["counts"] == [[1, 0, 0], [0, 0, 0], [0, 0, 0]], custom_action
none_action = dispatch.evaluation_action(
    __import__("numpy").asarray(dispatch_input["observation"], dtype="float32"),
    "none",
    dispatch_input["options"],
    dispatch_input["project_dir"],
)
assert none_action == 0, none_action
print("Python algorithm boundary: OK")
