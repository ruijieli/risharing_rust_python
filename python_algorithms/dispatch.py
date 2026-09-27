"""User-editable empty-vehicle dispatch policies."""

import numpy as np

_MODEL_CACHE = {}

# Register each dispatch algorithm's Gymnasium action-space type here.
# New matrix policies use "continuous"; scalar target-grid policies use "discrete".
ACTION_MODES = {
    "none": "none",
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


def _raw_action(state: np.ndarray, algorithm: str,
                model_paths: dict[str, str]) -> dict:
    """Return the algorithm-native action before integer-flow conversion."""
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
    return {"kind": "proportions", "values": np.asarray(action).tolist()}


def _proportions_to_counts(state: np.ndarray, values) -> list[list[int]]:
    """Project row weights to integer vehicle flows with largest remainders."""
    matrix = np.asarray(values, dtype=np.float64)
    size = state.size
    if matrix.shape != (size, size):
        raise ValueError(f"dispatch matrix shape {matrix.shape}, expected {(size, size)}")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("dispatch matrix contains invalid values")

    counts = np.zeros((size, size), dtype=np.int64)
    for origin, supply_value in enumerate(state):
        supply = int(round(float(supply_value)))
        if supply == 0:
            continue
        total = float(matrix[origin].sum())
        if total <= np.finfo(np.float64).eps:
            counts[origin, origin] = supply
            continue
        quotas = supply * matrix[origin] / total
        base = np.floor(quotas).astype(np.int64)
        counts[origin] = base
        remaining = supply - int(base.sum())
        order = np.argsort(-(quotas - base), kind="stable")
        counts[origin, order[:remaining]] += 1
    return counts.tolist()


def _target_to_counts(state: np.ndarray, value: int) -> list[list[int]]:
    size = state.size
    if not 1 <= value <= size:
        raise ValueError(f"discrete dispatch action {value} exceeds action space")
    counts = np.zeros((size, size), dtype=np.int64)
    counts[:, value - 1] = np.rint(state).astype(np.int64)
    return counts.tolist()


def _to_core_decision(state: np.ndarray, action: dict) -> dict:
    """Convert one algorithm-native action to the integer Rust contract."""
    if action["kind"] == "none" or (
        action["kind"] == "discrete" and action["value"] == 0
    ):
        return {"kind": "none"}
    if action["kind"] == "proportions":
        counts = _proportions_to_counts(state, action["values"])
    elif action["kind"] == "discrete":
        counts = _target_to_counts(state, action["value"])
    else:
        raise ValueError(f"Unknown raw dispatch action kind: {action['kind']}")
    return {"kind": "flow_matrix", "counts": counts}


def choose_action(observation: list[float], algorithm: str,
                  model_paths: dict[str, str]) -> dict:
    """Return only the two Rust-core actions: none or integer flow_matrix."""
    state = np.asarray(observation, dtype=np.float32)
    return _to_core_decision(state, _raw_action(state, algorithm, model_paths))


def training_action_to_decision(observation: np.ndarray, algorithm: str,
                                action) -> dict:
    """Adapt an RL library action to the same integer Rust contract."""
    state = np.asarray(observation, dtype=np.float32)
    mode = action_mode(algorithm)
    if mode == "none":
        raw = {"kind": "none"}
    elif mode == "continuous":
        raw = {"kind": "proportions", "values": np.asarray(action).tolist()}
    elif mode == "discrete":
        raw = {"kind": "discrete", "value": int(np.asarray(action).item())}
    else:
        raise ValueError(f"Unknown action mode: {mode}")
    return _to_core_decision(state, raw)


def evaluation_action(observation: np.ndarray, algorithm: str,
                      model_paths: dict[str, str]):
    """Convert the common dispatch decision into a Gymnasium-compatible action."""
    decision = _raw_action(np.asarray(observation, dtype=np.float32), algorithm, model_paths)
    kind = decision["kind"]
    if kind == "none":
        return 0
    if kind == "proportions":
        return np.asarray(decision["values"], dtype=np.float32)
    if kind == "discrete":
        return int(decision["value"])
    raise ValueError(f"Unknown dispatch action kind: {kind}")
