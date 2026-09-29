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
env = RustRideSharingEnv("config.toml")
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(action)
```

- `observation`：长度为 `num_grids` 的 `float32` 数组，表示每个 H3 网格的空闲车辆数。
- 所有调度算法的动作：`num_grids × num_grids` 非负连续比例矩阵。
- Python算法输出连续比例矩阵，`action[a][b]` 表示从A区调往B区的相对权重。
- `reward`：该调度周期新增的成功服务订单数。
- `terminated`：仿真时间达到 `end_time_s`。
- `info`：包含 `served_passengers` 和 `current_time`。

周期边界先完成当前批次匹配，再把匹配后的实际空闲车辆数返回给策略。
策略输出比例矩阵后，Rust 使用最大余数法一次性转换为整数车辆流量，
保证每行之和等于对应区域的实际空闲车辆数。对角线表示车辆留在
原区域，不请求 OSRM。不再存在 Python 提前整数化或 Rust 二次缩放。

## 通用算法配置

Python 按选中的算法名读取同名配置表；Rust 不知道 `maximum/PPO/SAC`
等具体算法，也不知道模型路径或超参数：

```toml
[python.matching]
name = "maximum"
[python.matching.maximum]

[python.matching.nearest]
strategy = "passenger"
max_matches = 0

[python.dispatch]
name = "ppo"
[python.dispatch.ppo]
model_path = "models/ppo_ridesharing_model.zip"

[python.dispatch.sac]
model_path = "models/sac_ridesharing_model.zip"
```

例如 `name = "nearest"` 时读取 `[python.matching.nearest]`，
`name = "ppo"` 时读取 `[python.dispatch.ppo]`。相对路径以配置文件
所在目录为基准。

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
@register("my_dispatch", trainer=my_trainer)
def my_dispatch(observation, options, project_dir):
    return {"kind": "proportions", "values": values}
```

Python 不处理 H3、不读取车辆内部对象，也不请求 OSRM。

内置调度算法：

| 名称 | 含义 |
|---|---|
| `none` | 输出单位矩阵，所有车辆留在原区域 |
| `random` | 随机比例矩阵 |
| `ppo` | PPO 输出比例矩阵 |
| `sac` | SAC 输出比例矩阵 |

Python 在周期边界通过 `apply_matches()` 先提交匹配关系，然后把匹配后的
观测交给策略。连续比例矩阵通过 `advance(proportions)` 立即执行。
增加新算法时，在 `dispatch.py` 或 `matching.py`
中注册函数；需要训练时同时提供 `trainer`。训练、评估入口和 Rust 内核不需要修改。

## 事件顺序

```text
读取调度周期边界的新订单
  → Python 匹配，Rust 更新成功匹配车辆状态
  → 返回匹配后的空闲车观测
  → Python 策略输出连续比例矩阵
  → Rust 按实际空闲车数转换为整数流量并规划空车路径
  → 更新车辆状态和位置
  → 推进仿真时间
  → 返回 observation/reward/done/info
```

现阶段继续保留原业务语义：未匹配订单在本批结束后流失；`idle` 和 `empty_trip` 可以参与匹配；空车调度只操作 `idle` 车辆。
