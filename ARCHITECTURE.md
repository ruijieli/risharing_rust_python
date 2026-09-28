# 架构与算法接口

## 分层边界

```text
Python：训练、评估、匹配策略、空车调度策略
                  │ 匹配关系 / 调度动作
                  ▼
Rust：H3、离散事件、车辆状态、OSRM、奖励、可视化快照
```

算法开发者只需要接触：

- `python_algorithms/matching.py`
- `python_algorithms/dispatch.py`

`training/rust_env.py` 在每个小步调用学生编写的匹配算法，再把匹配关系和调度动作提交给 Rust。
Rust 不会嵌入或回调 Python。

Rust 内部进一步分层：

| 文件 | 职责 |
|---|---|
| `lib.rs` | 声明模块并注册 `_rust_core`，不放仿真业务 |
| `python_api.rs` | PyO3 参数/返回值转换，导出 `RustSimulation` |
| `environment.rs` | `reset/step/reward/done` 和可视化环境控制 |
| `simulator.rs` | 离散事件、匹配执行、车辆状态推进 |
| `grid.rs` | H3 网格、观测和调度动作映射 |
| `osrm.rs` | 路径服务请求和轨迹插值 |
| `model.rs` | 车辆、乘客、坐标等核心数据结构 |

## Gymnasium 接口

```python
env = RustRideSharingEnv("config.toml", action_mode="continuous")
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(action)
```

- `observation`：长度为 `num_grids` 的 `float32` 数组，表示每个 H3 网格的空闲车辆数。
- PPO原始动作：`num_grids × num_grids` 非负比例矩阵。
- DQN原始动作：`0` 表示不调度，`1..num_grids` 表示统一调往对应网格。
- Python算法最终输出整数车辆流量矩阵，`counts[a][b]` 表示从A区调往B区的车辆数。
- `reward`：该调度周期新增的成功服务订单数。
- `terminated`：仿真时间达到 `end_time_s`。
- `info`：包含 `served_passengers` 和 `current_time`。

比例到整数流量的转换位于Python `dispatch.py`，使用最大余数法，保证每一行
之和等于该区域空闲车辆数。对角线表示车辆留在原区域，不请求OSRM。
Rust不理解PPO比例或DQN离散动作，只验证并执行最终整数矩阵。
一个周期内仍保持“先匹配、后空车调度”。如果匹配消耗了策略观测中的部分空闲车，
Rust会按原整数流量比例压缩该行到剩余车辆数，并再次用最大余数法保持守恒。

## 通用算法配置

Python 读取算法选择与不透明的 `options`；Rust 不知道 `maximum/PPO/DQN/TRPO`
等具体名称，也不知道模型路径或超参数：

```toml
[python.matching]
name = "maximum"
[python.matching.options]

[python.dispatch]
name = "ppo"
[python.dispatch.options]
model_path = "models/ppo_ridesharing_model.zip"
```

`options` 由对应 Python 算法自行解释。相对路径以配置文件所在目录为基准。

## 匹配算法接口

在 `matching.py` 中修改：

```python
@register("my_matching")
def my_matching(cars, passengers, *, threshold_m, options, project_dir):
    return [[car_id, passenger_id], ...]
```

车辆和乘客均为包含 `id/lat/lon` 的字典。Python 只返回匹配关系；OSRM、车辆状态和订单统计由 Rust 管理。

## 空车调度接口

在 `dispatch.py` 中修改：

```python
@register("my_dispatch", action_mode="continuous", trainer=my_trainer)
def my_dispatch(observation, options, project_dir):
    return {"kind": "proportions", "values": values}
```

也可以返回：

```python
{"kind": "none"}
{"kind": "flow_matrix", "counts": [[...], ...]}
```

Python 不处理 H3、不读取车辆内部对象，也不请求 OSRM。

内置调度算法：

| 名称 | 动作模式 | 含义 |
|---|---|---|
| `none` | none | 不进行空车调度，用作基准组 |
| `random` | continuous | 随机比例经转换后得到整数流量 |
| `ppo` | continuous | PPO比例矩阵经转换后得到整数流量 |
| `dqn` | discrete | DQN目标网格经转换后得到整数流量 |

Python 每个小步先调用 `matching.match_vehicles()`，再通过 `RustSimulation.step(matches, counts)`
提交匹配关系和可选的整数流量矩阵。增加新算法时，在 `dispatch.py` 或 `matching.py`
中注册函数；需要训练时同时提供 `trainer`。训练、评估入口和 Rust 内核不需要修改。

## 事件顺序

```text
读取本批新订单
  → Python 匹配
  → Rust 批量规划接驾与送客路径
  → 在调度周期边界将整数流量转换为空车路径
  → 更新车辆状态和位置
  → 推进仿真时间
  → 返回 observation/reward/done/info
```

现阶段继续保留原业务语义：未匹配订单在本批结束后流失；`idle` 和 `empty_trip` 可以参与匹配；空车调度只操作 `idle` 车辆。
