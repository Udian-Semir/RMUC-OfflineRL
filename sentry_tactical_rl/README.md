# 单哨兵战术 RL Demo

这是 [Sentry_radar_side_decision.md](../Sentry_radar_side_decision.md) 的第一版可运行骨架。它是**雷达站侧**的战术决策组件：策略看到语义地图、雷达式实体状态和每个候选目标的路径代价，选择 `goal`、攻击目标和交战模式；哨兵端的导航只负责验证和执行目标。

当前是纯 Python 的低保真 2D 战术环境，用于验证接口和训练链路。RMUC 2026 原始黑白障碍图、`0.35 m` 哨兵膨胀 A* 图、彩色语义标注图和 `semantic_map_aligned.json` 已作为第一版静态地图输入交付；像素对应可用 `python -m sentry_tactical_rl.tools.validate_semantic_map` 校验。加载器统一按 JSON 声明的下左角世界坐标存储障碍、目标锚点和语义多边形。主哨兵和蓝方哨兵使用膨胀图，英雄/工程/步兵使用原始物理图，空中单位不受地面障碍约束。正式部署前仍需完成真实雷达坐标校准和精确增益触发区标定。`SemanticMap.demo()` 仍仅用于冒烟测试，不能替代正式地图。

## 当前完成度

已实现：语义地图/硬禁区、动态威胁与路径代价、候选 goal 生成、雷达可见性遮罩、`(goal, target, fire_mode)` 动作、反应式敌我陪练、PPO、动作 mask、checkpoint、4 子进程并行环境，以及到既有 Gazebo 战术/导航协议的 JSON 边界。60 s 后蓝方前哨使用前方过道约 7 m 的固定合法射击位 `(9.35, 10.65)m`；由于前哨为高目标，该指定点允许专用射界，不改变车辆交战的普通 LOS 规则。Aerial 明确选择建筑时会切到稳定空中攻击位，并忽略地面障碍射线；车辆/建筑连续停留和火控规则仍然生效。若当前一秒的合法入射伤害足以击穿哨兵 HP，则优先锁定最大伤害威胁单位。并行 worker 使用独立 seed，栅格观测通过共享内存传递，GAE 按各局 `done` 边界独立计算。`match_rules.py` 已实现基地 `5000 HP + 150` 虚拟护盾、前哨 `1500 HP`、前哨存活时基地无敌、累计基地失血重建机会、300 秒重建截止、60/180/300 秒阶段状态，以及官方终局比较顺序。当前环境已经以真实建筑伤害替代“在前哨区域内站人就扣血”的旧逻辑。`sparring_adapter.py` 会从环境快照构造与 offlineRL 训练完全同序的 161 维向量；环境已补齐双方六角色固定槽位，红方哨兵之外的 11 个单位可用冻结 IQL checkpoint 每 5 秒生成子目标，并由 A*/射线/弹药/热量检查执行。

暂未实现：正式场地地图的坐标/通行校准、真实命中盒/弹道、经济与兑换、完整等级/经验与弱化状态、能量机关、完整姿态系统、堡垒储备弹药和有序隧道触发、雷达输入桥接、ROS2 发布节点、批量/向量化陪练推理、多个风格 checkpoint 的策略池，以及实车标定。地面单位目前会在阵亡读条后于己方补给区以 10% HP 和 30 秒无敌回归；空中单位只能攻击、不能被攻击。普通有效 DPS 单位的目标锁定伤害在 3 m/5 m 圈内按 `140/70 HP/s` 结算，哨兵为 `180/100 HP/s`；建筑为 `200 HP/s`，还必须满足目标意图、射程、视线和连续停留，避免把路过前哨错当推塔。英雄为 `200 HP/4 s`。它们仍是待实测标定的 2D 战术近似，不得将 offline 陪练模式的 PPO 曲线视为正式比赛实力。

### 雷达侧传统代价计算

`radar_costmap.py` 是 RL 之外的传统算法层：它读取 ROS 静态地图，按 `0.35 m` 欧氏净空膨胀障碍生成硬禁区，融合雷达敌方 track 为威胁场，并输出候选 goal 的可达性和 A* 路径代价。窄通道使用障碍距离 cost 选择安全中线；宽阔区域保留所有满足净空的位置。RL 不应自己猜测“能不能过狗洞/起伏路”。

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

## 地图校准

当前 A* 已对交付的 0.1 m 哨兵膨胀栅格验证基地、前哨、四条 tunnel 和全部语义锚点的连通性；这只说明 PNG 投影后的离散地图没有自相矛盾，**不**证明实车可通过。膨胀图可用下面命令确定性重建：

```bash
python3.10 -m sentry_tactical_rl.tools.export_inflated_astar_map \
  --json sentry_tactical_rl/assets/semantic_map_aligned.json \
  --obstacle-map sentry_tactical_rl/assets/blackwhite_map.png \
  --clearance-m 0.35 \
  --out sentry_tactical_rl/assets/blackwhite_astar_inflated_0p35m.png
```

原始图用于物理障碍和射线，膨胀图只用于哨兵中心的 goal/A*；实车本地 ESDF 仍需做最后安全投影。先填写
`sentry_tactical_rl/assets/radar_landmark_template.json` 中四个 `world_xy_m`：它们必须是同一物理中心在雷达/裁判 `map` 坐标系的实测位置，不要填 Foxglove 预览读数。

```bash
~/miniconda3/envs/nerfstudio/bin/python \
  -m sentry_tactical_rl.tools.calibrate_semantic_map \
  --landmarks sentry_tactical_rl/assets/radar_landmark_template.json \
  --model affine
```

该命令生成 `semantic_map_calibration.json`，其中记录像素到 `map` 的矩阵和逐地标残差。至少使用 4 个分散且不共线的点；初步要求最大残差不超过 `0.2 m`。若残差呈现位置相关的系统偏差，改用 `--model homography` 并补充地标，不要用手工偏移硬凑。

正式 PPO 之前还必须实测并写入规则/代价模型：狗洞/坡段/起伏路的实际通行性、基地至前哨及跨场路线的实车耗时、射线遮挡与可交战距离、命中/伤害/热量/弹药参数，以及雷达位置/速度/血量的延迟和误差分布。当前项目按已绘制的黑白可通行区作为哨兵硬约束，不另做哨兵外廓膨胀。

## 本机训练状态

本轮使用系统 `python3` 的 CPU PyTorch 和 4 个并行环境：每个环境采样 32 步，每次更新合计 128 样本；60 次更新耗时约 1005 秒，主进程峰值 RSS 约 1.27 GB。RTX 5070 可见，但该解释器的 torch 为 CPU build；切换 CUDA 环境前必须重新跑并行冒烟测试，不能只根据 `nvidia-smi` 判断训练已经在 GPU 上。

## 快速运行

```bash
python3.10 -m sentry_tactical_rl.smoke
~/miniconda3/envs/nerfstudio/bin/python -m unittest discover -s tests -v
python3.10 -m sentry_tactical_rl.train --config sentry_tactical_rl/configs/demo.yaml
# 4 个独立比赛环境并行采样，32 x 4 = 128 samples/update
python3 -m sentry_tactical_rl.train \
  --config sentry_tactical_rl/configs/online_ppo_hp20_inflated035_parallel4.yaml \
  --map-json sentry_tactical_rl/assets/semantic_map_aligned.json
# 训练时实时查看 reward / cost / 伤害 / 目标切换曲线
python3.10 -m sentry_tactical_rl.train --config sentry_tactical_rl/configs/demo.yaml --live
# 使用已交付的 RMUC 2026 语义图和黑白障碍图，并单独保存结果
python3.10 -m sentry_tactical_rl.train \
  --config sentry_tactical_rl/configs/demo.yaml \
  --map-json sentry_tactical_rl/assets/semantic_map_aligned.json \
  --obstacle-map sentry_tactical_rl/assets/blackwhite_map.png \
  --out-dir runs/sentry_tactical_rmuc2026 --live
# 使用完整 roster 与冻结 IQL 陪练（当前是未校准的战术仿真，不是正式结论）
python3.10 -m sentry_tactical_rl.train \
  --config sentry_tactical_rl/configs/offline_sparring.yaml
```

## Database Sentry Hybrid Replay

Replay one official database sentry track as the red ego while all other 11
slots run the frozen offlineRL sparring policies.  The result is explicitly a
hybrid diagnostic: the sentry state comes from the official SQLite record;
companions and building interactions come from the tactical simulator.

```bash
python3.10 -m sentry_tactical_rl.tools.replay_database_sentry_sparring \
  --game-id 1778631047459 --camp red \
  --profile aggressive --live
```

It writes `replay.gif`, `final.png`, `replay.csv`, and `summary.json` under
`runs/database_sentry_sparring_<game_id>_<camp>/`. Press `Esc` to stop a live
OpenCV window early; omit `--live` to generate only the files.

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

当前 demo 使用离散锚点。规则层和 A* 投影后的 `goal_xy` 会作为当前路线终点锁存，直到哨兵进入默认 `0.35 m` 到达容差；移动地面 target 在哨兵行驶期间只更新 target 显示并排队下一追击点，不会反复替换当前路线。哨兵到达后，下一个决策周期才按 target 新位置计算射击站位和切换路线；这种到点后的追击点更新不计入 PPO 的 goal-switch 惩罚。低血量回补给区和 60 秒后仍存活的敌方前哨命令可以抢占普通路线，无法规划的 goal 会释放锁存以避免死锁。

回放中的橙色十字是规则层和 A* 投影后真正准备发布的 goal，不是网络原始 anchor。顶部第三行逐秒显示 `goal=(x,y)m`、最终 `target` 和 `fire`；同样的数据写入 `steps.csv` 的 `published_goal_x_m / published_goal_y_m / published_target / published_fire` 字段。

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
