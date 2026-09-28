"""Train a registered dispatch algorithm using the Rust simulation core."""

import argparse
import sys
from pathlib import Path

# Support both `python -m training.train` and VS Code's direct-file Run button.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    parser.add_argument("--algorithm")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    project = PROJECT_ROOT
    config_path = (project / args.config).resolve()
    config = load_config(config_path)
    training = config["training"]
    algorithm = args.algorithm or training["algorithm"]
    options = training.get("options", {})
    algorithm_dir = (config_path.parent / config["python"]["algorithm_dir"]).resolve()
    if str(algorithm_dir) not in sys.path:
        sys.path.insert(0, str(algorithm_dir))
    import dispatch

    env = RustRideSharingEnv(config_path, dispatch_algorithm=algorithm)
    check_env(env, warn=True)
    if args.check_only:
        print(f"Rust RL environment OK: algorithm={algorithm}, grids={env.num_grids}")
        return

    model, timesteps, output = dispatch.create_training_model(
        algorithm, env, options, config_path.parent
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model.learn(total_timesteps=timesteps)
    model.save(output)
    print(f"训练完成：{output}.zip")


if __name__ == "__main__":
    main()
