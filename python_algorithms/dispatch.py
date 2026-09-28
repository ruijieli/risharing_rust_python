"""User-editable empty-vehicle dispatch algorithm registry."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

_MODEL_CACHE = {}
_ALGORITHMS: dict[str, dict] = {}


def register(name: str, *, action_mode: str, trainer: Callable | None = None):
    """Register inference and optional training hooks for one algorithm."""
    if action_mode not in {"none", "continuous", "discrete"}:
        raise ValueError(f"Invalid action mode: {action_mode}")

    def decorator(predictor: Callable) -> Callable:
        if name in _ALGORITHMS:
            raise ValueError(f"Dispatch algorithm already registered: {name}")
        _ALGORITHMS[name] = {
            "action_mode": action_mode,
            "predict": predictor,
            "trainer": trainer,
        }
        return predictor
    return decorator


def available_algorithms() -> tuple[str, ...]:
    return tuple(sorted(_ALGORITHMS))


def _algorithm(name: str) -> dict:
    try:
        return _ALGORITHMS[name]
    except KeyError as error:
        available = ", ".join(available_algorithms())
        raise ValueError(f"Unknown dispatch algorithm: {name}; available: {available}") from error


def action_mode(algorithm: str) -> str:
    return _algorithm(algorithm)["action_mode"]


def _resolve_path(value: str, project_dir: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(project_dir) / path


def _model_path(options: dict, project_dir: str | Path) -> str:
    try:
        return str(_resolve_path(options["model_path"], project_dir))
    except KeyError as error:
        raise ValueError("dispatch options must contain 'model_path'") from error


def _load_sb3_model(algorithm: str, path: str):
    key = (algorithm, path)
    if key not in _MODEL_CACHE:
        if algorithm == "ppo":
            from stable_baselines3 import PPO
            model_class = PPO
        elif algorithm == "dqn":
            from stable_baselines3 import DQN
            model_class = DQN
        else:
            raise ValueError(f"No built-in model loader for: {algorithm}")
        _MODEL_CACHE[key] = model_class.load(path)
    return _MODEL_CACHE[key]


def _train_ppo(env, options: dict, project_dir: str | Path):
    from stable_baselines3 import PPO
    model = PPO(
        "MlpPolicy", env, verbose=int(options.get("verbose", 1)),
        n_steps=int(options["n_steps"]), batch_size=int(options["batch_size"]),
        learning_rate=float(options["learning_rate"]),
        clip_range=float(options["clip_range"]), ent_coef=float(options["ent_coef"]),
    )
    return model, int(options["total_timesteps"]), _resolve_path(options["output_path"], project_dir)


def _train_dqn(env, options: dict, project_dir: str | Path):
    from stable_baselines3 import DQN
    model = DQN(
        "MlpPolicy", env, verbose=int(options.get("verbose", 1)),
        learning_rate=float(options["learning_rate"]),
        buffer_size=int(options["buffer_size"]),
        learning_starts=int(options["learning_starts"]),
        batch_size=int(options["batch_size"]),
    )
    return model, int(options["total_timesteps"]), _resolve_path(options["output_path"], project_dir)


@register("none", action_mode="none")
def _predict_none(state: np.ndarray, options: dict, project_dir: str | Path) -> dict:
    del state, options, project_dir
    return {"kind": "none"}


@register("random", action_mode="continuous")
def _predict_random(state: np.ndarray, options: dict, project_dir: str | Path) -> dict:
    del options, project_dir
    values = np.random.uniform(0, 1, size=(state.size, state.size)).astype(np.float32)
    return {"kind": "proportions", "values": values.tolist()}


@register("ppo", action_mode="continuous", trainer=_train_ppo)
def _predict_ppo(state: np.ndarray, options: dict, project_dir: str | Path) -> dict:
    model = _load_sb3_model("ppo", _model_path(options, project_dir))
    action, _ = model.predict(state, deterministic=bool(options.get("deterministic", True)))
    return {"kind": "proportions", "values": np.asarray(action).tolist()}


@register("dqn", action_mode="discrete", trainer=_train_dqn)
def _predict_dqn(state: np.ndarray, options: dict, project_dir: str | Path) -> dict:
    model = _load_sb3_model("dqn", _model_path(options, project_dir))
    action, _ = model.predict(state, deterministic=bool(options.get("deterministic", True)))
    return {"kind": "discrete", "value": int(np.asarray(action).item())}


def _raw_action(state: np.ndarray, algorithm: str, options: dict,
                project_dir: str | Path = ".") -> dict:
    return _algorithm(algorithm)["predict"](state, options, project_dir)


def _proportions_to_counts(state: np.ndarray, values) -> list[list[int]]:
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
    if action["kind"] == "flow_matrix":
        return action
    if action["kind"] == "none" or (action["kind"] == "discrete" and action["value"] == 0):
        return {"kind": "none"}
    if action["kind"] == "proportions":
        counts = _proportions_to_counts(state, action["values"])
    elif action["kind"] == "discrete":
        counts = _target_to_counts(state, action["value"])
    else:
        raise ValueError(f"Unknown raw dispatch action kind: {action['kind']}")
    return {"kind": "flow_matrix", "counts": counts}


def choose_action(observation: list[float], algorithm: str, options: dict | None = None,
                  project_dir: str | Path = ".") -> dict:
    state = np.asarray(observation, dtype=np.float32)
    return _to_core_decision(state, _raw_action(state, algorithm, options or {}, project_dir))


def training_action_to_decision(observation: np.ndarray, algorithm: str, action) -> dict:
    state = np.asarray(observation, dtype=np.float32)
    mode = action_mode(algorithm)
    if mode == "none":
        raw = {"kind": "none"}
    elif mode == "continuous":
        raw = {"kind": "proportions", "values": np.asarray(action).tolist()}
    else:
        raw = {"kind": "discrete", "value": int(np.asarray(action).item())}
    return _to_core_decision(state, raw)


def evaluation_action(observation: np.ndarray, algorithm: str, options: dict | None = None,
                      project_dir: str | Path = "."):
    decision = _raw_action(np.asarray(observation, dtype=np.float32), algorithm,
                           options or {}, project_dir)
    if decision["kind"] == "none":
        return 0
    if decision["kind"] == "proportions":
        return np.asarray(decision["values"], dtype=np.float32)
    if decision["kind"] == "discrete":
        return int(decision["value"])
    raise ValueError(f"Unknown dispatch action kind: {decision['kind']}")


def create_training_model(algorithm: str, env, options: dict,
                          project_dir: str | Path = "."):
    trainer = _algorithm(algorithm)["trainer"]
    if trainer is None:
        raise ValueError(f"Dispatch algorithm does not provide a trainer: {algorithm}")
    return trainer(env, options, project_dir)
