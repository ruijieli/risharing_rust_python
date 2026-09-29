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

    def __init__(self, config_path: str | Path):
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
        import matching
        self.dispatch = dispatch
        self.matching = matching
        matching_config = config["python"]["matching"]
        self.matching_algorithm = matching_config["name"]
        self.matching_options = matching_config.get(self.matching_algorithm, {})
        self.project_dir = self.config_path.parent
        self.micro_steps = max(
            1,
            int(config["simulation"]["dispatch_cycle_s"])
            // int(config["simulation"]["batch_interval_s"]),
        )
        self.core = RustSimulation(str(self.config_path))
        self.num_grids = int(self.core.num_grids())
        self.action_space = spaces.Box(
            0.0, 1.0, shape=(self.num_grids, self.num_grids), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            0.0, np.inf, shape=(self.num_grids,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.core.reset(seed)
        self._apply_current_matching()
        self._observation = np.asarray(self.core.observation(), dtype=np.float32)
        return self._observation.copy(), {}

    def _matching_pairs(self):
        payload = json.loads(self.core.matching_input_json())
        matches = self.matching.match_vehicles(
            payload["cars"],
            payload["passengers"],
            float(payload["threshold_m"]),
            self.matching_algorithm,
            self.matching_options,
            str(self.project_dir),
        )
        return [tuple(pair) for pair in matches]

    def _apply_current_matching(self) -> float:
        return float(self.core.apply_matches(self._matching_pairs()))

    def step(self, action):
        proportions = self.dispatch.training_action_to_proportions(action, self.num_grids)
        observation, _unused, terminated, truncated, info_json = self.core.advance(
            proportions.tolist()
        )
        reward = 0.0
        for _ in range(self.micro_steps - 1):
            if terminated or truncated:
                break
            observation, micro_reward, terminated, truncated, info_json = self.core.step(
                self._matching_pairs()
            )
            reward += float(micro_reward)
        if not (terminated or truncated):
            reward += self._apply_current_matching()
            observation = self.core.observation()
        self._observation = np.asarray(observation, dtype=np.float32)
        info = json.loads(info_json)
        info["served_passengers"] = self.core.served_passengers()
        return (self._observation.copy(), reward,
                bool(terminated), bool(truncated), info)

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
