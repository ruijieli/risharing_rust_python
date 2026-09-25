"""Train PPO or DQN using the Rust simulation core and the unified TOML config."""

import argparse
import sys
from pathlib import Path

# Support both `python -m training.train` and VS Code's direct-file Run button.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env

from training.rust_env import RustRideSharingEnv


def load_config(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with path.open("rb") as handle:
        return tomllib.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--algorithm", choices=("ppo", "dqn"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    project = PROJECT_ROOT
    config_path = (project / args.config).resolve()
    config = load_config(config_path)
    training = config["training"]
    algorithm = args.algorithm or training["algorithm"]
    mode = "continuous" if algorithm == "ppo" else "discrete"
    env = RustRideSharingEnv(config_path, action_mode=mode)
    check_env(env, warn=True)
    if args.check_only:
        print(f"Rust RL environment OK: algorithm={algorithm}, grids={env.num_grids}")
        return

    models = project / "models"
    models.mkdir(exist_ok=True)
    if algorithm == "ppo":
        model = PPO(
            "MlpPolicy", env, verbose=1,
            n_steps=int(training["ppo_n_steps"]),
            batch_size=int(training["ppo_batch_size"]),
            learning_rate=float(training["ppo_learning_rate"]),
            clip_range=float(training["ppo_clip_range"]),
            ent_coef=float(training["ppo_ent_coef"]),
        )
        timesteps = int(training["total_timesteps"])
        output = models / "ppo_ridesharing_model"
    else:
        model = DQN(
            "MlpPolicy", env, verbose=1,
            learning_rate=float(training["dqn_learning_rate"]),
            buffer_size=int(training["dqn_buffer_size"]),
            learning_starts=int(training["dqn_learning_starts"]),
            batch_size=int(training["dqn_batch_size"]),
        )
        timesteps = int(training["dqn_total_timesteps"])
        output = models / "dqn_ridesharing_model"
    model.learn(total_timesteps=timesteps)
    model.save(output)
    print(f"训练完成：{output}.zip")


if __name__ == "__main__":
    main()
