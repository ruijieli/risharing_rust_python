#!/usr/bin/env python3
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "python_algorithms"))

import api
import dispatch

matching_input = {
    "cars": [{"id": "car_0", "lat": 30.66, "lon": 104.06}],
    "passengers": [{"id": "pax_0", "lat": 30.661, "lon": 104.061}],
}
matches = json.loads(api.match_json(json.dumps(matching_input), "maximum", 1000.0))
assert matches == [["car_0", "pax_0"]], matches

dispatch_input = {
    "observation": [1.0, 0.0, 0.0],
    "model_paths": {"ppo": "unused", "dqn": "unused"},
}
action = json.loads(api.dispatch_action_json(json.dumps(dispatch_input), "random"))
assert action["kind"] == "matrix" and len(action["values"]) == 3, action
assert dispatch.action_mode("none") == "continuous"
assert dispatch.action_mode("ppo") == "continuous"
assert dispatch.action_mode("dqn") == "discrete"
none_action = dispatch.evaluation_action(
    __import__("numpy").asarray(dispatch_input["observation"], dtype="float32"),
    "none",
    dispatch_input["model_paths"],
)
assert none_action.shape == (3, 3) and not none_action.any(), none_action
print("Python algorithm boundary: OK")
