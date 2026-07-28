# 单哨兵战术 RL Demo

这是 [SENTRY_RL_PLAN.md](../SENTRY_RL_PLAN.md) 的第一版可运行骨架。它是**雷达站侧**的战术决策组件：策略看到语义地图、雷达式实体状态和每个候选目标的路径代价，选择 `goal`、攻击目标和交战模式；哨兵端的导航只负责验证和执行目标。

当前是纯 Python 的低保真 2D 战术环境，用于验证接口和训练链路。RMUC 2026 黑白障碍图、彩色语义标注图和 `semantic_map_aligned.json` 已作为第一版静态地图输入交付；像素对应可用 `python -m sentry_tactical_rl.tools.validate_semantic_map` 校验。加载器现在统一按 JSON 声明的下左角世界坐标存储障碍、目标锚点和语义多边形，避免它们在栅格中上下镜像。正式训练前仍需完成真实雷达坐标校准、哨兵专属通行掩码和精确增益触发区标定。`SemanticMap.demo()` 仍仅用于冒烟测试，不能替代正式地图。

## 当前完成度

已实现：语义地图/硬禁区、动态威胁与路径代价、候选 goal 生成、雷达可见性遮罩、`(goal, target, fire_mode)` 动作、反应式敌我陪练、PPO、动作 mask、checkpoint，以及到既有 Gazebo 战术/导航协议的 JSON 边界。`match_rules.py` 已实现基地 `5000 HP + 150 虚拟护盾`、前哨 `1500 HP`、前哨存活时基地无敌、累计基地失血重建机会、300 秒重建截止、60/180/300 秒阶段状态，以及官方终局比较顺序。当前环境已经以真实建筑伤害替代“在前哨区域内站人就扣血”的旧逻辑。`sparring_adapter.py` 会从环境快照构造与 offlineRL 训练完全同序的 161 维向量，并已用冻结的蓝方哨兵 IQL checkpoint 推理验证。

暂未实现：正式场地地图的坐标/通行校准、真实命中盒/弹道、经济与兑换、复活、能量机关、完整姿态系统、堡垒储备弹药和有序隧道触发、雷达输入桥接、ROS2 发布节点、全 6 兵种 roster 和 offlineRL 子目标执行器、以及实车标定。现有离线模型现在可以对显式环境快照推理，但还不能直接替换脚本陪练：当前环境只实例化部分角色，且其 5 秒子目标仍必须经过导航/射击安全层。

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
~/miniconda3/envs/nerfstudio/bin/python -m unittest discover -s tests -v
python3.10 -m sentry_tactical_rl.train --config sentry_tactical_rl/configs/demo.yaml
# 训练时实时查看 reward / cost / 伤害 / 目标切换曲线
python3.10 -m sentry_tactical_rl.train --config sentry_tactical_rl/configs/demo.yaml --live
# 使用已交付的 RMUC 2026 语义图和黑白障碍图，并单独保存结果
python3.10 -m sentry_tactical_rl.train \
  --config sentry_tactical_rl/configs/demo.yaml \
  --map-json sentry_tactical_rl/assets/semantic_map_aligned.json \
  --obstacle-map sentry_tactical_rl/assets/blackwhite_map.png \
  --out-dir runs/sentry_tactical_rmuc2026 --live
```

训练依赖 `numpy`、`torch` 和可选的 `matplotlib` 实时窗口。本工作区默认 `python` 是没有 torch 的 Python 3.13；请使用装有 torch 的 Python 3.10 或团队的 conda 环境。即使不加 `--live`，默认 checkpoint 和 `metrics.csv` 仍会写入 `runs/sentry_tactical_demo/`；加上 `--live` 后还会持续刷新 `training_live.png`。

## Foxglove 语义图预览

下面的本项目内启动脚本发布黑白占据图和 JSON 多边形轮廓，默认 topic 为
`/sentry/semantic_map`（`OccupancyGrid`）和 `/sentry/semantic_regions`
（`MarkerArray`），不会占用或修改外部导航的 `/map`。它固定使用
`ROS_DOMAIN_ID=42` 和 Foxglove bridge 端口 `8766`，避免混入已有机器人或 Gazebo
常用的 domain `0`：

```bash
cd ~/workspace/RMUC-OfflineRL
bash sentry_tactical_rl/tools/run_foxglove_semantic_preview.sh
```

Foxglove 连接 `ws://<本机局域网 IP>:8766` 后，在 3D 面板中启用
`/sentry/semantic_map` 和 `/sentry/semantic_regions`。预览节点会发布
`map -> semantic_preview` 的恒等静态变换，因此 3D 面板保持 `Display frame = map`
即可显示两者。该变换只连接可视化坐标系，不会修改外部导航的 `/map` 或 TF。若需要改变
domain，可将目标 domain 作为首个参数，例如
`bash sentry_tactical_rl/tools/run_foxglove_semantic_preview.sh 42`。默认使用独立的
`semantic_preview` frame，仅用于像素对齐检查；它不会冒充尚未完成校准的真实导航
`map` 坐标。可用 `--frame-id map` 强制发布到 `map`，但应在地标校准完成后才这样做。
预览栅格的左下角默认与 `map` 原点重合 `(0, 0)`；多边形会同步平移，所以仍和底图对齐。
若实测坐标系需要额外偏移，可加 `--origin-x <x> --origin-y <y>`。加载时会自动裁掉源 PNG
完全透明的外框，避免 Foxglove 将其显示为灰色未知区域。

不要把 Foxglove 读到的预览坐标直接写入 `polygon_xy_m`：裁剪偏移和显示用的 y 比例会使
它和 JSON 坐标不同。在 3D 面板使用 `Publish point` 点击地图（`Display frame = map`），
预览脚本终端会打印可直接填入 JSON 的 `polygon_xy_m: [x, y]`。

## 当前接口

当前 demo 的每个战术决策为：

```text
(goal_anchor_id, target_id, fire_mode)
```

- `goal_anchor_id`：从语义地图生成的可达候选区域中选择；
- `target_id`：三名敌方陪练、敌方前哨站或 `none`；敌方基地的规则状态已存在，但第一版 PPO action mask 仍禁止其直接选为目标；
- `fire_mode`：保持停火或在安全、可见、射程内交战。

每个候选点都由传统 A* 计算路径总代价和可达性。仿真中不允许穿过硬禁区；真实部署时应由现有导航栈接管该职责。

当前 demo 使用离散锚点；追击时不再使用全局 5 秒硬保持，以允许每秒更新目标。连续 `goal_xy`、目标变化率限制、3 m 交战环和短时域候选重排仍需在动作执行器阶段实现。

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
