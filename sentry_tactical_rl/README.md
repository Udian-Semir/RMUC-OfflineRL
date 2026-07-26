# 单哨兵战术 RL Demo

这是 [SENTRY_RL_PLAN.md](../SENTRY_RL_PLAN.md) 的第一版可运行骨架。它是**雷达站侧**的战术决策组件：策略看到语义地图、雷达式实体状态和每个候选目标的路径代价，选择 `goal`、攻击目标和交战模式；哨兵端的导航只负责验证和执行目标。

当前是纯 Python 的低保真 2D 战术环境，用于验证接口和训练链路。RMUC 2026 黑白障碍图、彩色语义标注图和 `semantic_map_aligned.json` 已作为第一版静态地图输入交付；像素对应可用 `python -m sentry_tactical_rl.tools.validate_semantic_map` 校验。正式训练前仍需完成雷达坐标校准、哨兵专属通行掩码、精确增益触发区和规则状态转移校验。`SemanticMap.demo()` 仍仅用于冒烟测试，不能替代正式地图。

## 当前完成度

已实现：语义地图/硬禁区、动态威胁与路径代价、候选 goal 生成、雷达可见性遮罩、`(goal, target, fire_mode)` 动作、反应式敌我陪练、PPO、动作 mask、checkpoint，以及到既有 Gazebo 战术/导航协议的 JSON 边界。

暂未实现：正式场地地图的坐标/通行校准、真实裁判伤害与资源规则、雷达输入桥接、ROS2 发布节点、BC/IQL/DT 陪练适配和实车标定。现有离线模型不能直接塞进本环境：它们依赖 161 维裁判日志观测，必须先定义“仿真状态/雷达状态 → 该观测”的一致转换，避免无意引入全知信息。

### 雷达侧传统代价计算

`radar_costmap.py` 是 RL 之外的传统算法层：它读取 ROS 静态地图，按哨兵半径膨胀障碍生成硬禁区，融合雷达敌方 track 为威胁场，并输出候选 goal 的可达性和 A* 路径代价。RL 不应自己猜测“能不能过狗洞/起伏路”。

`radar_features.py` 进一步将每个语义锚点的 `reachable / path_cost / path_length / mean_threat` 和多通道代价栅格构造成策略输入。这就是后续真实雷达状态接入 PPO 推理模型的前置层。

本机现有可用底图：`~/workspace/Gazebo_simulation_for_sentry/src/sentry_perception/config/RMUC2025.yaml`，对应 28×15 m、0.01 m 的 `RMUC2025.png`。它目前只提供静态占据栅格；增益点、战术点、哨兵专属通行性和地图坐标核对仍需补充。**该文件名明确为 RMUC2025，只能用于验证雷达代价图管线，不能直接作为 RMUC2026 的最终训练地图。**

Gazebo 工程另有标注为 RMUC 2026 的碰撞网格 `rmuc2026_map.dae`。可用下面脚本导出供人工标注的俯视底图：

```bash
~/miniconda3/envs/nerfstudio/bin/python \
  sentry_tactical_rl/tools/extract_2026_mesh_map.py \
  --mesh ~/workspace/Gazebo_simulation_for_sentry/src/sentry_sim/models/rmuc_map/meshes/rmuc2026_map.dae
```

导出的网格外缘约为 `29.05 × 16.05 m`，与裁判日志的 `28 × 15 m` 不完全相同。它可能包含墙体/外缘；先用红方基地、蓝方基地和前哨站三个地标求出到雷达/裁判坐标的变换，再把标注坐标接入训练。导出的图是**结构标注底图**，不是经验证的可通行黑白地图；坡面、狗洞和边缘必须按哨兵外廓和实车规则单独标注为可通行或硬禁区。

```bash
python3.10 -m sentry_tactical_rl.costmap_smoke \
  --map-yaml ~/workspace/Gazebo_simulation_for_sentry/src/sentry_perception/config/RMUC2025.yaml
```

默认以 0.10 m 生成雷达侧代价图，避免粗下采样封死真实窄通道；`sentry_radius_m`、坡段通行规则和地图原点必须用实车尺寸与联调结果校准。

## 本机训练状态

本机 RTX 5070 Laptop GPU（8 GB）已用 CUDA PyTorch 跑通 5 次 PPO 更新，约 6.5 秒；当前不需要 AutoDL。请使用 `~/miniconda3/envs/nerfstudio/bin/python`（CUDA）运行训练；系统默认 Python 3.13 没有安装 torch。

## 快速运行

```bash
python3.10 -m sentry_tactical_rl.smoke
python3.10 -m sentry_tactical_rl.train --config sentry_tactical_rl/configs/demo.yaml
```

训练只依赖仓库已有的 `numpy` 和 `torch`。本工作区默认 `python` 是没有 torch 的 Python 3.13；请使用装有 torch 的 Python 3.10 或团队的 conda 环境。默认 checkpoint 会写入 `runs/sentry_tactical_demo/`。

## 当前接口

每个战术决策为：

```text
(goal_anchor_id, target_id, fire_mode)
```

- `goal_anchor_id`：从语义地图生成的可达候选区域中选择；
- `target_id`：三名敌方陪练之一，或 `none`；
- `fire_mode`：保持停火或在安全、可见、射程内交战。

每个候选点都由传统 A* 计算路径总代价和可达性。仿真中不允许穿过硬禁区；真实部署时应由现有导航栈接管该职责。

## 与 Gazebo 导航工程的边界

此 demo 的 `GridNavigationBackend` 仅用于训练仿真。真实集成时，决策层只需要把已验证的目标点发布到现有导航工程的：

```text
/navigation/goal_pose  (PoseStamped, frame_id=map)
```

现有导航链路会继续完成 JPS/全局路径、MINCO、局部 ESDF/MPC 和 `cmd_vel` 仲裁。此项目不直接发布底盘速度。

`deployment.py` 还提供受 schema 约束的战术动作 JSON；字段与 Gazebo 工程的 `RadarTacticalState` JSON 协议对齐。将来桥接节点只应把已验证的 goal/action 转发给 CDC/ROS2，不能允许该 demo 绕过本地硬安全。

## 下一步需要替换的输入

1. 正式场地静态地图、狗洞/起伏路与哨兵专属硬禁区；可从 `configs/arena_example.yaml` 的格式开始；
2. 战术锚点、增益区、射线/掩体和软代价层；
3. 雷达实际可获得的实体状态与可见性/置信度；
4. 真实运动、交战、热量、资源和裁判规则的仿真参数；
5. 来自本仓库 BC/IQL/DT 和手写策略组成的陪练池。

LLM 不在本 demo 中。
