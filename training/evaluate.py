"""Evaluate a trained model through the Rust environment with live visualization."""

import argparse
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.rust_env import RustRideSharingEnv


def load_config(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with path.open("rb") as handle:
        return tomllib.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估模型并实时显示车辆位置")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument(
        "--algorithm",
        help="调度算法名称；默认读取 config.toml 的 [python].dispatch",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-keep-open", action="store_true")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_config(config_path)
    python_config = config["python"]
    algorithm = args.algorithm or python_config["dispatch"]
    algorithm_dir = (PROJECT_ROOT / python_config["algorithm_dir"]).resolve()
    if str(algorithm_dir) not in sys.path:
        sys.path.insert(0, str(algorithm_dir))
    import dispatch

    action_mode = dispatch.action_mode(algorithm) # 返回动作是连续的还是离散的
    model_paths = {
        key.removesuffix("_model"): str((PROJECT_ROOT / value).resolve())
        for key, value in python_config.items()
        if key.endswith("_model")
    }

    env = RustRideSharingEnv(config_path, action_mode=action_mode)
    observation, _ = env.reset()
    print(f"调度算法：{algorithm}；动作模式：{action_mode}")
    visualization_enabled = bool(config["simulation"]["visualization"])
    if visualization_enabled:
        url = env.start_visualization()
        print(f"可视化：{url}")
        if not args.no_browser:
            webbrowser.open(url)
    else:
        print("可视化已关闭；本次评估不会启动网页服务。")

    delay = (
        float(config["simulation"]["visualization_step_delay_ms"]) / 1000.0
        if visualization_enabled
        else 0.0
    )
    total_reward = 0.0
    steps = 0
    started = time.perf_counter()
    while True:
        action = dispatch.evaluation_action(observation, algorithm, model_paths)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if delay > 0:
            time.sleep(delay)
        if terminated or truncated:
            break

    elapsed = time.perf_counter() - started
    print(
        f"评估完成：steps={steps}, served={int(total_reward)}, "
        f"wall_time={elapsed:.3f}s, steps_per_second={steps / max(elapsed, 1e-9):.2f}"
    )
    if (
        visualization_enabled
        and config["simulation"]["keep_visualization_alive"]
        and not args.no_keep_open
    ):
        print("页面将保持打开；在终端按 Ctrl+C 退出。")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
