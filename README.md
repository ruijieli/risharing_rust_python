# 成都网约车离散事件仿真：Rust 内核 + Python 算法

这是 `ridesharing_chengdu-main` 的独立重构版本，原 Python 工程不会被修改。

## 架构边界

Rust 负责：

- 离散事件循环与仿真时间；
- 基于 `h3o` 的 H3 网格、车辆到网格映射和可视化 GeoJSON；
- 车辆四态状态机；
- CSV 订单加载和结果输出；
- OSRM 路径请求、并发控制和轨迹点插值；
- 实时快照和 `8000` 可视化服务。
- 强化学习环境的观测、动作验证与执行、状态推进、奖励和 episode 时间。

Python 只负责：

- `python_algorithms/matching.py`：派单匹配；
- `python_algorithms/dispatch.py`：空车调度，包括 PPO/DQN 推理。
- `training/rust_env.py`：很薄的 Gymnasium/Stable-Baselines3 适配器；
- `training/train.py`：Stable-Baselines3 的 PPO/DQN 训练入口。
- `training/evaluate.py`：模型评估、测速和实时可视化入口。

Python 调度算法只接收 Rust 生成的网格观测并返回动作矩阵或离散动作，不再
处理 H3、车辆坐标或抽样执行。后续新增算法主要编辑 `python_algorithms`，
内核数据结构不需要修改。

## 前置服务

- 成都 OSRM：`http://127.0.0.1:5001`
- MapLibre 矢量瓦片：`http://127.0.0.1:8081`
- Conda 环境 `RL`，其中已安装 NumPy、NetworkX 和 Stable-Baselines3。

## 首次构建

```bash
cd /Users/liuqinghua/Documents/Codex/TS_REVISE/ridesharing_chengdu_rust_main
conda activate RL
export PYO3_PYTHON="$(which python)"
cargo build --release
```

首次构建需要从 crates.io 下载 Rust 依赖，以后可以离线增量构建。
构建脚本会将当前 Conda Python 的动态库目录写入 macOS 二进制的
`rpath`，因此构建前必须先激活最终运行时使用的 Python 环境。

## 运行

```bash
conda activate RL
export PYO3_PYTHON="$(which python)"
cargo run --release -- config.toml
```

也可以使用一键脚本：

```bash
./scripts/run.sh
```

打开 `http://127.0.0.1:8000`。仿真完成后，若
`keep_visualization_alive = true`，页面继续保留，终端按 `Ctrl+C` 退出。

这个 Rust 二进制入口用于无 Gymnasium 的批处理仿真。算法开发者通常不需要使用它。

## 强化学习训练

训练与一次性仿真共用同一个 Rust 内核、`config.toml`、订单数据和模型目录，
新工程不再读取 `ridesharing_chengdu-main`。

训练前必须确认成都 OSRM 正常：

```bash
curl "http://127.0.0.1:5001/nearest/v1/driving/104.0648,30.6543"
```

训练 PPO：

```bash
conda activate RL
./scripts/train.sh ppo config.toml
```

训练 DQN：

```bash
./scripts/train.sh dqn config.toml
```

训练脚本会自动构建 Python 可调用的 Rust 扩展，模型分别保存到：

```text
models/ppo_ridesharing_model.zip
models/dqn_ridesharing_model.zip
```

所有训练参数都集中在 `config.toml` 的 `[training]`。

## 评估、测速与可视化

```bash
conda activate RL
./scripts/evaluate.sh
```

它调用同一个 Rust Gymnasium 环境，自动加载模型、启动
`http://127.0.0.1:8000` 并实时发布车辆位置。也可以在 VS Code 的“运行和调试”
中选择“评估 PPO 并可视化（Rust 环境）”。算法开发者不需要运行 `main.rs`
或手动使用 Cargo。

评估入口默认读取 `config.toml` 的 `[python].dispatch`，支持：

```toml
dispatch = "none"    # 无调度基准
dispatch = "random"  # 随机调度基准
dispatch = "ppo"     # PPO 调度
dispatch = "dqn"     # DQN 调度
```

也可以临时覆盖配置，例如 `./scripts/evaluate.sh none config.toml`。新增调度算法
只需在 `python_algorithms/dispatch.py` 注册和实现，不需要修改 Rust。

## 快速验证算法接口

```bash
conda activate RL
python scripts/check_python_algorithms.py
```

第一次建议先检查 20 辆车、100 条订单的小规模 Gymnasium 环境：

```bash
python -m training.train --config config.smoke.toml --algorithm ppo --check-only
```

## 配置

正式运行参数集中在 `config.toml`，小规模验证参数集中在
`config.smoke.toml`。正式配置默认使用：

```toml
num_cars = 500
sample_size = 3000
dispatch = "ppo"
```

## 当前兼容语义

- 每个批次只匹配本批新订单，未匹配订单立即流失；
- `idle` 和 `empty_trip` 车辆都允许参与匹配；
- 接驾成功后进入 `en_route_to_pickup`，随后进入 `on_trip`；
- 空车调度只选择 `idle` 车辆；
- 时间统计仍按离散步数累加，与现有 Python 版本一致。
