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

Rust 源码中，`lib.rs` 只注册模块；Python 接口位于 `python_api.rs`，Gymnasium
环境语义位于 `environment.rs`，具体仿真业务位于 `simulator.rs`。Rust 只作为
Python 扩展被调用，不会回调 Python。

Python 只负责：

- `python_algorithms/matching.py`：派单匹配；
- `python_algorithms/dispatch.py`：注册空车调度的推理与训练实现；
- `training/rust_env.py`：很薄的 Gymnasium/Stable-Baselines3 适配器；
- `training/train.py`：调用已注册 trainer 的通用训练入口；
- `training/evaluate.py`：模型评估、测速和实时可视化入口。

Python 负责在每个小步计算车辆匹配，并将匹配关系与调度动作提交给 Rust。
调度算法接收周期边界匹配后的网格观测，统一输出连续比例矩阵。
Rust 再根据实际空闲车辆数转换成整数流量。后续新增算法只需编辑
`python_algorithms`。

## 前置服务

- 成都 OSRM：`http://127.0.0.1:5001`
- MapLibre 矢量瓦片：`http://127.0.0.1:8081`
- Conda 环境 `RL`，其中已安装 NumPy、NetworkX 和 Stable-Baselines3。

## 首次构建

学生不需要手动编译或运行 Rust。首次执行 `./scripts/train.sh` 或
`./scripts/evaluate.sh` 时，脚本会自动构建 `_rust_core` Python 扩展。

## 强化学习训练

训练与一次性仿真共用同一个 Rust 内核、`config.toml`、订单数据和模型目录，
新工程不再读取 `ridesharing_chengdu-main`。

训练前必须确认成都 OSRM 正常：

```bash
curl "http://127.0.0.1:5001/nearest/v1/driving/104.0648,30.6543"
```

训练配置中的算法：

```bash
conda activate RL
./scripts/train.sh "" config.toml
```

也可临时指定算法，程序会自动读取同名配置表，例如 `[training.sac]`：

```bash
./scripts/train.sh sac config.toml
```

训练脚本会自动构建 Python 可调用的 Rust 扩展，输出位置由
`[training.<algorithm>].output_path` 指定，例如：

```text
models/ppo_ridesharing_model.zip
```

PPO 和 SAC 参数分别位于 `[training.ppo]` 和 `[training.sac]`，切换
`[training].algorithm` 或命令行算法后会自动选择对应配置。

## 评估、测速与可视化

```bash
conda activate RL
./scripts/evaluate.sh
```

它调用同一个 Rust Gymnasium 环境，自动加载模型、启动
`http://127.0.0.1:8000` 并实时发布车辆位置。也可以在 VS Code 的“运行和调试”
中选择“评估 PPO 并可视化（Rust 环境）”。算法开发者不需要运行 `main.rs`
或手动使用 Cargo。

评估入口默认读取 `config.toml` 的 `[python.dispatch].name`。例如：

```toml
[python.dispatch]
name = "ppo"

[python.dispatch.ppo]
model_path = "models/ppo_ridesharing_model.zip"
deterministic = true
```

也可以临时覆盖配置，例如 `./scripts/evaluate.sh none config.toml`。新增调度算法
只需在 `python_algorithms/dispatch.py` 注册和实现，不需要修改训练/评估入口或
Rust。新增 matching 算法同理，只需在 `matching.py` 注册实现。

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
[python.dispatch]
name = "ppo"
```

## 当前兼容语义

- 每个批次只匹配本批新订单，未匹配订单立即流失；
- `idle` 和 `empty_trip` 车辆都允许参与匹配；
- 接驾成功后进入 `en_route_to_pickup`，随后进入 `on_trip`；
- 空车调度只选择 `idle` 车辆；
- 时间统计仍按离散步数累加，与现有 Python 版本一致。
