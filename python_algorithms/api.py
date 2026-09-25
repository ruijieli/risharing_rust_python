"""Stable JSON boundary between Rust and user-editable Python algorithms."""

import json

from dispatch import choose_action
from matching import match_vehicles


def match_json(payload_json: str, algorithm: str, threshold_m: float) -> str:
    payload = json.loads(payload_json)
    result = match_vehicles(
        payload["cars"], payload["passengers"], float(threshold_m), algorithm
    )
    return json.dumps(result)


def dispatch_action_json(payload_json: str, algorithm: str) -> str:
    payload = json.loads(payload_json)
    result = choose_action(
        payload["observation"],
        algorithm,
        payload["model_paths"],
    )
    return json.dumps(result)
