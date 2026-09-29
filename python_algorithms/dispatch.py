"""User-editable empty-vehicle dispatch algorithm registry."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

_MODEL_CACHE = {}
_ALGORITHMS: dict[str, dict] = {}


def register(name: str, *, trainer: Callable | None = None):
    """Register inference and optional training hooks for one algorithm."""
    def decorator(predictor: Callable) -> Callable:
        if name in _ALGORITHMS:
            raise ValueError(f"Dispatch algorithm already registered: {name}")
        _ALGORITHMS[name] = {
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
        elif algorithm == "sac":
            from stable_baselines3 import SAC
            model_class = SAC
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


def _train_sac(env, options: dict, project_dir: str | Path):
    from stable_baselines3 import SAC
    model = SAC(
        "MlpPolicy", env, verbose=int(options.get("verbose", 1)),
        learning_rate=float(options["learning_rate"]),
        buffer_size=int(options["buffer_size"]),
        learning_starts=int(options["learning_starts"]),
        batch_size=int(options["batch_size"]),
        tau=float(options.get("tau", 0.005)),
        gamma=float(options.get("gamma", 0.99)),
        train_freq=int(options.get("train_freq", 1)),
        gradient_steps=int(options.get("gradient_steps", 1)),
    )
    return model, int(options["total_timesteps"]), _resolve_path(options["output_path"], project_dir)


@register("none")
def _predict_none(state: np.ndarray, options: dict, project_dir: str | Path) -> dict:
    del options, project_dir
    return {"kind": "proportions", "values": np.eye(state.size, dtype=np.float32).tolist()}


@register("random")
def _predict_random(state: np.ndarray, options: dict, project_dir: str | Path) -> dict:
    del options, project_dir
    values = np.random.uniform(0, 1, size=(state.size, state.size)).astype(np.float32)
    return {"kind": "proportions", "values": values.tolist()}


@register("ppo", trainer=_train_ppo)
def _predict_ppo(state: np.ndarray, options: dict, project_dir: str | Path) -> dict:
    model = _load_sb3_model("ppo", _model_path(options, project_dir))
    action, _ = model.predict(state, deterministic=bool(options.get("deterministic", True)))
    return {"kind": "proportions", "values": np.asarray(action).tolist()}


@register("sac", trainer=_train_sac)
def _predict_sac(state: np.ndarray, options: dict, project_dir: str | Path) -> dict:
    model = _load_sb3_model("sac", _model_path(options, project_dir))
    action, _ = model.predict(state, deterministic=bool(options.get("deterministic", True)))
    return {"kind": "proportions", "values": np.asarray(action).tolist()}


def _raw_action(state: np.ndarray, algorithm: str, options: dict,
                project_dir: str | Path = ".") -> dict:
    return _algorithm(algorithm)["predict"](state, options, project_dir)


def _proportion_matrix(values, size: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != (size, size):
        raise ValueError(f"dispatch matrix shape {matrix.shape}, expected {(size, size)}")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("dispatch matrix contains invalid values")
    return matrix


def choose_action(observation: list[float], algorithm: str, options: dict | None = None,
                  project_dir: str | Path = ".") -> dict:
    state = np.asarray(observation, dtype=np.float32)
    decision = _raw_action(state, algorithm, options or {}, project_dir)
    if decision.get("kind") != "proportions":
        raise ValueError(f"Unknown dispatch action kind: {decision.get('kind')}")
    return {
        "kind": "proportions",
        "values": _proportion_matrix(decision["values"], state.size).tolist(),
    }


def training_action_to_proportions(action, num_grids: int) -> np.ndarray:
    return _proportion_matrix(action, num_grids)


def evaluation_action(observation: np.ndarray, algorithm: str, options: dict | None = None,
                      project_dir: str | Path = "."):
    decision = _raw_action(np.asarray(observation, dtype=np.float32), algorithm,
                           options or {}, project_dir)
    if decision.get("kind") != "proportions":
        raise ValueError(f"Unknown dispatch action kind: {decision.get('kind')}")
    return _proportion_matrix(decision["values"], np.asarray(observation).size)


def create_training_model(algorithm: str, env, options: dict,
                          project_dir: str | Path = "."):
    trainer = _algorithm(algorithm)["trainer"]
    if trainer is None:
        raise ValueError(f"Dispatch algorithm does not provide a trainer: {algorithm}")
    return trainer(env, options, project_dir)
