"""Thin Gymnasium adapter; environment state, H3 and transitions live in Rust."""

import json
import sys
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from _rust_core import RustSimulation


class RustRideSharingEnv(gym.Env):
    """RL environment using Rust for matching, routing and vehicle transitions."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config_path: str | Path,
                 dispatch_algorithm: str = "ppo"):
        super().__init__()
        self.config_path = Path(config_path).resolve()
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib
        with self.config_path.open("rb") as handle:
            config = tomllib.load(handle)
        algorithm_dir = (self.config_path.parent / config["python"]["algorithm_dir"]).resolve()
        if str(algorithm_dir) not in sys.path:
            sys.path.insert(0, str(algorithm_dir))
        import dispatch
        self.dispatch = dispatch
        self.dispatch_algorithm = dispatch_algorithm
        self.action_mode = dispatch.action_mode(dispatch_algorithm)
        self.core = RustSimulation(str(self.config_path))
        self.num_grids = int(self.core.num_grids())
        if self.action_mode == "none":
            self.action_space = spaces.Discrete(1)
        elif self.action_mode == "continuous":
            self.action_space = spaces.Box(
                0.0, 1.0, shape=(self.num_grids, self.num_grids), dtype=np.float32
            )
        elif self.action_mode == "discrete":
            self.action_space = spaces.Discrete(self.num_grids + 1)
        else:
            raise ValueError("action_mode must be none, continuous or discrete")
        self.observation_space = spaces.Box(
            0.0, np.inf, shape=(self.num_grids,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._observation = np.asarray(self.core.reset(seed), dtype=np.float32)
        return self._observation.copy(), {}

    def step(self, action):
        decision = self.dispatch.training_action_to_decision(
            self._observation, self.dispatch_algorithm, action
        )
        if decision["kind"] == "none":
            result = self.core.step_none()
        else:
            result = self.core.step_flow_matrix(decision["counts"])
        observation, reward, terminated, truncated, info_json = result
        self._observation = np.asarray(observation, dtype=np.float32)
        return (self._observation.copy(), float(reward),
                bool(terminated), bool(truncated), json.loads(info_json))

    def start_visualization(self) -> str:
        """Start the Rust-owned web server and return its URL."""
        return self.core.start_visualization()

    @property
    def total_passengers(self) -> int:
        """Number of passengers loaded for the current evaluation episode."""
        return int(self.core.total_passengers())

    @property
    def served_passengers(self) -> int:
        """Cumulative number of passengers served in the current episode."""
        return int(self.core.served_passengers())

    def render(self):
        print(f"idle vehicles by grid: {self.core.observation()}")
