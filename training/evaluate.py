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
        help="调度算法名称；默认读取 config.toml 的 [python.dispatch].name",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-keep-open", action="store_true")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_config(config_path)
    python_config = config["python"]
    dispatch_config = python_config["dispatch"]
    algorithm = args.algorithm or dispatch_config["name"]
    options = dispatch_config.get(algorithm, {})
    algorithm_dir = (config_path.parent / python_config["algorithm_dir"]).resolve()
    if str(algorithm_dir) not in sys.path:
        sys.path.insert(0, str(algorithm_dir))
    import dispatch

    env = RustRideSharingEnv(config_path)
    observation, _ = env.reset()
    print(f"调度算法：{algorithm}；动作：连续比例矩阵")
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
    steps = 0
    started = time.perf_counter()
    while True:
        action = dispatch.evaluation_action(
            observation, algorithm, options, config_path.parent
        )
        observation, _reward, terminated, truncated, info = env.step(action)
        steps += 1
        if delay > 0:
            time.sleep(delay)
        if terminated or truncated:
            break

    elapsed = time.perf_counter() - started
    total_passengers = env.total_passengers
    served_passengers = env.served_passengers
    service_rate = (
        served_passengers / total_passengers * 100.0
        if total_passengers
        else 0.0
    )
    print(
        f"评估完成：总乘客数={total_passengers}，"
        f"已服务人数/总乘客数={served_passengers}/{total_passengers}，"
        f"服务率={service_rate:.2f}%，steps={steps}，"
        f"wall_time={elapsed:.3f}s，"
        f"steps_per_second={steps / max(elapsed, 1e-9):.2f}"
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
