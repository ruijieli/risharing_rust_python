"""User-editable empty-vehicle dispatch policies."""

import numpy as np

_MODEL_CACHE = {}

# Register each dispatch algorithm's Gymnasium action-space type here.
# New matrix policies use "continuous"; scalar target-grid policies use "discrete".
ACTION_MODES = {
    "none": "continuous",
    "random": "continuous",
    "ppo": "continuous",
    "dqn": "discrete",
}


def action_mode(algorithm: str) -> str:
    """Return the Gymnasium action mode required by a dispatch algorithm."""
    try:
        return ACTION_MODES[algorithm]
    except KeyError as error:
        available = ", ".join(sorted(ACTION_MODES))
        raise ValueError(
            f"Unknown dispatch algorithm: {algorithm}; available: {available}"
        ) from error


def _load_model(algorithm, path):
    key = (algorithm, path)
    if key not in _MODEL_CACHE:
        if algorithm == "ppo":
            from stable_baselines3 import PPO
            _MODEL_CACHE[key] = PPO.load(path)
        elif algorithm == "dqn":
            from stable_baselines3 import DQN
            _MODEL_CACHE[key] = DQN.load(path)
    return _MODEL_CACHE.get(key)


def choose_action(observation: list[float], algorithm: str,
                  model_paths: dict[str, str]) -> dict:
    """Return ``none``, a grid-to-grid matrix, or a discrete grid action."""
    state = np.asarray(observation, dtype=np.float32)
    size = state.size
    if algorithm == "none":
        return {"kind": "none"}

    if algorithm == "random":
        action = np.random.uniform(0, 1, size=(size, size)).astype(np.float32)
    elif algorithm == "ppo":
        model = _load_model("ppo", model_paths["ppo"])
        action, _ = model.predict(state, deterministic=True)
    elif algorithm == "dqn":
        model = _load_model("dqn", model_paths["dqn"])
        discrete, _ = model.predict(state, deterministic=True)
        return {"kind": "discrete", "value": int(np.asarray(discrete).item())}
    else:
        raise ValueError(f"Unknown dispatch algorithm: {algorithm}")
    return {"kind": "matrix", "values": np.asarray(action).tolist()}


def evaluation_action(observation: np.ndarray, algorithm: str,
                      model_paths: dict[str, str]):
    """Convert the common dispatch decision into a Gymnasium-compatible action."""
    decision = choose_action(observation.tolist(), algorithm, model_paths)
    kind = decision["kind"]
    if kind == "none":
        size = observation.size
        return np.zeros((size, size), dtype=np.float32)
    if kind == "matrix":
        return np.asarray(decision["values"], dtype=np.float32)
    if kind == "discrete":
        return int(decision["value"])
    raise ValueError(f"Unknown dispatch action kind: {kind}")
