# 架构与算法接口

## 分层边界

```text
Python：训练、评估、匹配策略、空车调度策略
                  │ observation / action
                  ▼
Rust：H3、离散事件、车辆状态、OSRM、奖励、可视化快照
```

算法开发者只需要接触：

- `python_algorithms/matching.py`
- `python_algorithms/dispatch.py`
- `training/train.py`（更换或配置训练算法时）
- `training/evaluate.py`（评估入口）

`python_algorithms/api.py` 是稳定的 Rust/Python JSON 边界，不放业务算法。
`training/rust_env.py` 是很薄的 Gymnasium 适配器，不实现第二套仿真逻辑。

## Gymnasium 接口

```python
env = RustRideSharingEnv("config.toml", action_mode="continuous")
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(action)
```

- `observation`：长度为 `num_grids` 的 `float32` 数组，表示每个 H3 网格的空闲车辆数。
- PPO 动作：`num_grids × num_grids` 非负矩阵，行是车辆当前网格，列是目标网格。
- DQN 动作：`0` 表示不调度，`1..num_grids` 表示统一调往对应网格。
- `reward`：该调度周期新增的成功服务订单数。
- `terminated`：仿真时间达到 `end_time_s`。
- `info`：包含 `served_passengers` 和 `current_time`。

动作的检查、归一化抽样、网格中心转换和车辆调度均在 Rust 执行。

## 匹配算法接口

在 `matching.py` 中修改：

```python
def match_vehicles(cars, passengers, threshold_m, algorithm):
    return [[car_id, passenger_id], ...]
```

车辆和乘客均为包含 `id/lat/lon` 的字典。Python 只返回匹配关系；OSRM、车辆状态和订单统计由 Rust 管理。

## 空车调度接口

在 `dispatch.py` 中修改：

```python
def choose_action(observation, algorithm, model_paths):
    return {"kind": "matrix", "values": matrix}
```

也可以返回：

```python
{"kind": "none"}
{"kind": "discrete", "value": target}
```

Python 不处理 H3、不读取车辆内部对象，也不请求 OSRM。

内置调度算法：

| 名称 | 动作模式 | 含义 |
|---|---|---|
| `none` | continuous | 不进行空车调度，用作基准组 |
| `random` | continuous | 随机网格调度，用作基准组 |
| `ppo` | continuous | PPO 输出网格到网格动作矩阵 |
| `dqn` | discrete | DQN 输出单个目标网格 |

增加新算法时，在 `dispatch.py` 的 `ACTION_MODES` 注册名称和动作模式，并在
`choose_action()` 增加实现即可。训练、评估循环及 Rust 内核不需要修改。

## 事件顺序

```text
读取本批新订单
  → Python 匹配
  → Rust 批量规划接驾与送客路径
  → 到调度周期时执行 Python/RL 动作
  → Rust 批量规划空车路径
  → 更新车辆状态和位置
  → 推进仿真时间
  → 返回 observation/reward/done/info
```

现阶段继续保留原业务语义：未匹配订单在本批结束后流失；`idle` 和 `empty_trip` 可以参与匹配；空车调度只操作 `idle` 车辆。
