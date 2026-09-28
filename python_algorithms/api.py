"""Stable JSON boundary between Rust and user-editable Python algorithms."""

import json

from dispatch import choose_action
from matching import match_vehicles


def match_json(payload_json: str, algorithm: str) -> str:
    payload = json.loads(payload_json)
    result = match_vehicles(
        payload["cars"], payload["passengers"],
        float(payload["threshold_m"]), algorithm,
        payload.get("options", {}),
        payload.get("project_dir", "."),
    )
    return json.dumps(result)


def dispatch_action_json(payload_json: str, algorithm: str) -> str:
    payload = json.loads(payload_json)
    result = choose_action(
        payload["observation"],
        algorithm,
        payload.get("options", {}),
        payload.get("project_dir", "."),
    )
    return json.dumps(result)
