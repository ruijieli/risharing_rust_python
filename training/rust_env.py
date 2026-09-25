"""Thin Gymnasium adapter; environment state, H3 and transitions live in Rust."""

import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from _rust_core import RustSimulation


class RustRideSharingEnv(gym.Env):
    """RL environment using Rust for matching, routing and vehicle transitions."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config_path: str | Path, 
                 action_mode: str = "continuous"):
        super().__init__()
        self.config_path = Path(config_path).resolve()
        self.action_mode = action_mode
        self.core = RustSimulation(str(self.config_path))
        self.num_grids = int(self.core.num_grids())
        if action_mode == "continuous":
            self.action_space = spaces.Box(
                0.0, 1.0, shape=(self.num_grids, self.num_grids), dtype=np.float32
            )
        elif action_mode == "discrete":
            self.action_space = spaces.Discrete(self.num_grids + 1)
        else:
            raise ValueError("action_mode must be continuous or discrete")
        self.observation_space = spaces.Box(
            0.0, np.inf, shape=(self.num_grids,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.asarray(self.core.reset(seed), dtype=np.float32), {}

    def step(self, action):
        if self.action_mode == "continuous":
            result = self.core.step_continuous(np.asarray(action, dtype=np.float64).tolist())
        else:
            result = self.core.step_discrete(int(np.asarray(action).item()))
        observation, reward, terminated, truncated, info_json = result
        return (np.asarray(observation, dtype=np.float32), float(reward),
                bool(terminated), bool(truncated), json.loads(info_json))

    def start_visualization(self) -> str:
        """Start the Rust-owned web server and return its URL."""
        return self.core.start_visualization()

    def render(self):
        print(f"idle vehicles by grid: {self.core.observation()}")
