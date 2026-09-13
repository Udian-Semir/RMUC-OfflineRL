# simulation2D

`simulation2d` 是 RMUC-OfflineRL 仓库内的独立二维战术交互世界 ROS 2 Python 包。ROS 2
包名必须为小写，因此功能名称写作 `simulation2D`，`package.xml`、Python import 和
`ros2 run` 使用 `simulation2d`。

这个包从原 `RMUC-OfflineRL/sentry_tactical_rl` 拆分迁入，主要用于：

- 在 28 m x 15 m 场地中推进双方单位、建筑、血量、弹药、热量、死亡和复活状态；
- 让红方哨兵 PPO 与会移动、会受伤、会死亡的目标形成真实状态闭环；
- 构造 PPO 使用的 `15 x 150 x 280` 地图、171 维向量和动作 mask；
- 加载静态语义地图、哨兵膨胀 A* 地图并生成动态雷达威胁代价；
- 训练、评测和回放 PPO，或把高层动作转换为部署协议。

## 运行模式

### `scripted`：包内自包含交互

其他单位由世界模型的确定性/随机规则移动和交战，不需要比赛数据库、`rm_rl` 或冻结
checkpoint。它只用于不加载外部模型时的通信、地图和伤害闭环调试，不是默认实时模式。

### `offline`：冻结策略陪练

世界仍然是交互式的，但其他 11 个单位的高层意图来自外部 offlineRL checkpoint。
`agents/offline_sparring.py` 保留了适配器；运行该模式还需要原 `rm_rl` Python 包和对应
`rm_runs`/`sparring_style_pool` 陪练 checkpoint。默认红方 PPO checkpoint 已复制到本包的
`checkpoints/`，不再依赖 RMUC-OfflineRL 的绝对路径。实时节点和交互回放都默认使用
`offline`，与该 PPO checkpoint 的训练环境一致；SQLite 数据库和其他训练产物仍不复制。

## 快速使用

```bash
cd ~/workspace/RMUC-OfflineRL
colcon build --base-paths simulation2d --packages-select simulation2d --symlink-install
source /opt/ros/humble/setup.bash
source install/setup.bash

# 不加载 PyTorch/checkpoint，检查正式地图和世界状态闭环
ros2 run simulation2d simulation2d_smoke

# 实时运行世界并发布位置、goal、target、Gazebo 实际位姿、action、state 和 Marker
ros2 launch simulation2d simulation2d.launch.py

# 同时启动 simulation2d 自带 OpenCV 可视化
ros2 launch simulation2d simulation2d_visualizer.launch.py

# 默认已经加载包内 PPO checkpoint，直接运行即可
ros2 launch simulation2d simulation2d_visualizer.launch.py

# 用已有 PPO 按训练时的 offline 陪练配置回放
ros2 run simulation2d simulation2d_replay \
  --checkpoint /path/to/ppo_00155.pt \
  --steps 420

# 仅在不加载 rm_rl/checkpoint 的调试场景显式切换规则陪练
ros2 launch simulation2d simulation2d_visualizer.launch.py sparring_backend:=scripted

# 训练时默认加载包内正式静态地图和 demo/scripted 配置
ros2 run simulation2d simulation2d_train --updates 5

# 发布语义地图和区域轮廓供 Foxglove 查看
ros2 run simulation2d simulation2d_map_preview
```

PPO/回放/评测需要 PyTorch；ROS Humble 系统解释器 `/usr/bin/python3` 已包含当前工作区所需
的 `numpy/Pillow/OpenCV/PyYAML/PyTorch`。冻结陪练模式还需要外部 `rm_rl` 和 checkpoint。

## 目录总览

```text
simulation2d/
├── assets/          地图、语义标注和校准产物
├── checkpoints/     默认 PPO 部署模型
├── config/          世界和 PPO 实验配置
├── resource/        ament Python 包索引
├── simulation2d/    可导入 Python 源码
│   ├── agents/      非主哨兵策略适配
│   ├── mapping/     雷达代价地图和特征
│   ├── policy/      PPO 网络、采样和训练
│   ├── runtime/     部署动作边界
│   ├── tools/       回放、评测、地图工具
│   ├── visualization/ 渲染和预览
│   └── world/       交互世界核心
└── test/            包级回归测试
```

## 根目录文件

| 文件 | 功能 | 加载/安装内容 |
| --- | --- | --- |
| `package.xml` | ROS 2 包清单 | 声明 `ament_python`、ROS 消息、OpenCV、Pillow、NumPy、PyYAML 等运行依赖。 |
| `setup.py` | Python/ament 安装入口 | 安装所有子包、`assets/`、`config/`、README，并注册 smoke、训练、回放、评测、预览命令。 |
| `setup.cfg` | ROS 2 Python 可执行文件位置 | 将 console scripts 安装到 `lib/simulation2d`，供 `ros2 run` 查找。 |
| `README.md` | 包说明 | 说明运行模式、依赖、命令，以及每个目录和文件的职责。 |

## `checkpoints/` 默认模型

| 文件 | 功能/来源 |
| --- | --- |
| `sentry_ppo_00155.pt` | 默认实时 PPO；从 RMUC-OfflineRL 的 v2 固定评测最佳 checkpoint 原样复制，SHA-256 为 `df34ab8f646b392e1eacd9e2c28aa24e70e3f69e776a1edac86ed251429318ec`。构建后安装到 `share/simulation2d/checkpoints/`。 |

## 实时 ROS 接口

`simulation2d_node` 默认以 1 Hz 推进世界，并加载包内
`checkpoints/sentry_ppo_00155.pt`。该模型来自
`RMUC-OfflineRL/runs/sentry_ppo_hp20_inflated035_parallel4_v2/ppo_00155.pt`；固定
`seed=1001/1002 x aggressive/pressure/measured` 的 6-case 评测胜率为 `1.000`。后续 v3
候选在对应 6-case 评测中胜率为 `0.000`，因此没有采用“最新文件”作为默认模型。

`policy_mode=pursuit` 和 `sparring_backend=scripted` 仍可用于不加载 checkpoint 的通信回归；
一般运行不需要指定它们。

| Topic/Service | 类型 | 用途 |
| --- | --- | --- |
| `/simulation2d/battlefield/state` | `std_msgs/msg/String` JSON | 12 个单位完整槽位、连续坐标、HP、alive 和 respawn 状态；不依赖 Radar-Station 的 `interfaces`。 |
| `/rl/policy_goal_pose` | `geometry_msgs/msg/PoseStamped` | 当前执行 goal；Gazebo 的 `sentry_rl_goal_bridge` 已订阅该 topic。 |
| `/rl/tactical_action` | `std_msgs/msg/String` JSON | requested/executed goal、target、fire 和到点状态。 |
| `/simulation2d/state` | `std_msgs/msg/String` JSON | 回合、时间、reward、哨兵位置/HP、建筑 HP 和 outcome。 |
| `/simulation2d/selected_target_pose` | `geometry_msgs/msg/PoseStamped` | PPO 当前执行的 target 坐标；没有攻击目标时发布空 frame，并通过 label 发布 `TARGET NONE`。 |
| `/simulation2d/selected_target_label` | `std_msgs/msg/String` | target 的可读名称和 HP，例如 `TARGET blue hero HP 100/100`。 |
| `/simulation2d/gazebo_sentry_pose` | `geometry_msgs/msg/PoseStamped` | 从 TF `map -> base_footprint` 读取的 Gazebo 哨兵实际位置；TF 不可用时发布空 frame。 |
| `/simulation2d/gazebo_goal_feedback` | `std_msgs/msg/String` | `GAZEBO MOVING/ARRIVED`、到 goal 距离和 `/navigation/controller_status`。 |
| `/simulation2d/markers` | `visualization_msgs/msg/MarkerArray` | Foxglove/RViz 中显示单位、HP、PPO goal、target、Gazebo 实际哨兵和误差连线。 |
| `/simulation2d/reset` | `std_srvs/srv/Trigger` | 重置并进入下一 seed 的 episode。 |
| `/simulation2d/pause` | `std_srvs/srv/SetBool` | `true` 暂停，`false` 恢复。 |

simulation2d 是 RMUC-OfflineRL 下的独立 ROS 2 package，不修改 Radar-Station 的正式源码。
Gazebo 当前只消费 goal；`target_index/fire_mode` 仍通过 `/rl/tactical_action` 发布，后续
需要哨兵侧增加对应火控消费者。

### Goal/Target/Gazebo 可视化

`simulation2d_visualizer.launch.py` 会启动 package 自带的 OpenCV visualizer。图中和
`/simulation2d/markers` 使用同一套颜色语义：黄色菱形是 PPO goal，紫色圆环是 PPO target，
绿色十字/方块是 Gazebo 中通过 TF 回读的真实哨兵，橙色连线是 Gazebo 实际位置到 PPO goal
的误差；世界模型中的红/蓝单位仍使用红/蓝圆柱。右侧 roster 固定列出红蓝双方各 6 个
槽位、角色、坐标、HP 和 alive/respawn 状态；聚集单位使用真实位置点加放射展开标记，
不会因重叠看起来少兵种。左上角状态栏始终显示 `PPO GOAL`、`TARGET` 和
`GAZEBO MOVING/ARRIVED`，因此 target 为 `NONE` 或 Gazebo 尚未发布 TF 时也能区分“没有
目标”和“数据尚未到达”。

Gazebo 到位判定由 simulation2d 负责：实际 `map -> base_footprint` 距离当前 goal 不超过
`gazebo_goal_tolerance_m`（默认 `0.35 m`）即为 `ARRIVED`；同时接受
`/navigation/controller_status` 的 `goal_reached` 或 `goal_reached_position` 状态。新 goal、
episode reset 或控制器重新进入 tracking/recovery 会清除上一 goal 的到位状态。该判定只用于
反馈和可视化，不替代哨兵端行为树或局部导航的安全裁决。

当前导航路线会锁存到 `ARRIVED`。target 在途中继续实时更新位置和标记，但不会触发新 goal；
只有 Gazebo 几何到点或控制器报告到点后，下一次 PPO 决策才按 target 新位置发布下一条路线。

当 `map -> base_footprint` TF 可用时，节点在每次 PPO 推理前把 Gazebo 的连续哨兵位姿注入
世界模型；因此策略 observation、goal 到位判断和下一次 goal 选择使用真实哨兵位置，而不
是 2D 模型自己推进出来的位置。外部位姿模式仍保留世界模型的目标移动、受伤、击杀、复活
和 `simulate_fire` 伤害闭环，便于验证 target 是否真的会产生反馈。

## `simulation2d/` Python 包

| 文件 | 功能 | 加载内容 |
| --- | --- | --- |
| `__init__.py` | 公共入口 | 导出 `SentryTacticalEnv` 和 `MatchState`；不主动加载 PyTorch。 |
| `resources.py` | 资源路径解析 | 优先从 ament share 查找地图/配置；未安装时回退到源码目录。 |

## `simulation2d/world/` 交互世界核心

| 文件 | 功能 | 加载内容 |
| --- | --- | --- |
| `__init__.py` | world 公共接口 | 导出环境、单位、比赛状态和语义地图。 |
| `environment.py` | 主世界状态机 | 创建双方 12 个移动槽位和建筑，推进 1 Hz 动作、A* 移动、目标选择、开火、伤害、死亡、复活、补给、goal 锁存、reward 和 observation；`offline` 模式按需加载 `agents/offline_sparring.py`。 |
| `rules.py` | 比赛规则 | 加载/维护基地与前哨 HP、护盾、重建、比赛阶段、终局和胜负比较。 |
| `navigation.py` | 世界内导航后端 | 使用 `SemanticMap` 做 A* 路径查询并返回可达性、路径、成本和失败原因。 |
| `semantic_map.py` | 统一静态/语义地图 | 加载 aligned JSON、原始障碍 PNG、预膨胀 A* PNG、区域 polygon、语义 mask、目标锚点、LOS 和静态 cost。 |

## `simulation2d/mapping/` 雷达侧地图

| 文件 | 功能 | 加载内容 |
| --- | --- | --- |
| `__init__.py` | mapping 公共接口 | 导出 `RadarCostMap`、动态快照、RadarTrack 和特征构造器。 |
| `costmap.py` | 雷达静态/动态代价图 | 加载标准 ROS map YAML/PNG，按哨兵半径膨胀障碍；融合敌方 track 置信度、威胁权重和语义软 cost，提供 A* 查询。 |
| `features.py` | 雷达策略前处理 | 从 costmap、哨兵位置、语义 anchor 和敌方 track 生成 5 通道雷达图、goal 路径成本/威胁、goal mask 和 target 特征。该 5 通道格式是在线 adapter 的基础，不可直接冒充当前 PPO 的 15 通道输入。 |

## `simulation2d/agents/` 陪练适配

| 文件 | 功能 | 加载内容 |
| --- | --- | --- |
| `__init__.py` | agents 命名空间 | 保持可选陪练依赖与世界核心分离。 |
| `offline_sparring.py` | offlineRL 陪练适配器 | 将 `SentryTacticalEnv` 快照转换为原 offlineRL 161 维 observation，调用外部 `rm_rl.deploy.MLPPolicyRunner`，再把输出变成世界内 goal/target/fire 意图。仅 `sparring_backend=offline` 时加载。 |

## `simulation2d/policy/` PPO

| 文件 | 功能 | 加载内容 |
| --- | --- | --- |
| `__init__.py` | policy 命名空间 | 避免仅导入世界模型时就加载 PyTorch。 |
| `network.py` | Actor-Critic 网络 | 加载 15 通道地图 CNN、171 维向量 MLP、goal/target/fire 三个 masked action head 和 value head。 |
| `trainer.py` | PPO 算法实现 | 加载环境、网络和并行池，完成 rollout、GAE、裁剪损失、优化、checkpoint 保存/恢复和训练遥测。 |
| `parallel_env.py` | 多进程环境池 | Linux `fork` 启动独立世界，使用共享内存传输 map/vector/mask，按各局 done 边界采样。 |
| `train_cli.py` | 训练命令 | 加载 YAML、包内正式地图、PPO checkpoint 和设备配置；创建环境/训练器并保存训练指标与 checkpoint。 |

## `simulation2d/runtime/` 部署边界

| 文件 | 功能 | 加载内容 |
| --- | --- | --- |
| `__init__.py` | runtime 公共接口 | 导出 `TacticalDecision` 和 JSON 动作转换函数。 |
| `actions.py` | 高层动作序列化 | 将 goal/target/fire 转成雷达到哨兵的战术动作结构；不直接发布 `cmd_vel`，也不替代本地导航。 |
| `live_engine.py` | 实时世界驱动核心 | ROS 无关地封装 episode reset/step；支持包内 `PursuitPolicy` 和可选 `PpoPolicy`，便于测试和复用。 |
| `live_node.py` | 实时 ROS 2 节点 | 定时推进世界，发布完整 battlefield JSON、PoseStamped、action/state JSON 和 Marker；可回读 Gazebo TF 并注入 PPO。 |

## `simulation2d/visualization/` 可视化

| 文件 | 功能 | 加载内容 |
| --- | --- | --- |
| `__init__.py` | visualization 命名空间 | 将渲染依赖与核心世界隔离。 |
| `dashboard.py` | 训练指标面板 | 读取每次 PPO update 的 reward、损伤、cost、策略损失等指标，写 CSV，并可选生成实时曲线。 |
| `semantic_preview.py` | 语义地图预览数据 | 加载 aligned JSON/PNG，构造 OccupancyGrid 和语义 polygon 所需的无 ROS 中间表示。 |
| `replay_renderer.py` | 统一回放渲染 | 加载场地图、单位、建筑 HP、战斗事件和战术 action，生成逐帧图片及 MP4。 |
| `live_visualizer.py` | 实时 OpenCV 可视化 | 订阅 battlefield JSON、goal、target 和 Gazebo feedback；显示地图、双方 6+6 roster、聚集展开标记和到位状态。 |

## `simulation2d/tools/` 命令与离线工具

| 文件 | 功能 | 加载内容 |
| --- | --- | --- |
| `__init__.py` | tools 命名空间 | 供 module/console entry point 定位工具。 |
| `smoke.py` | 最小世界回归 | 加载包内 aligned JSON、原始障碍图和膨胀图，推进 12 个交互 step，检查 15 通道/171 维契约。 |
| `costmap_smoke.py` | ROS 地图代价检查 | 加载指定 ROS map YAML，膨胀障碍、添加动态威胁、跑 A* 并构造雷达特征。 |
| `replay_interactive.py` | PPO 交互回放 | 加载 PPO checkpoint、世界 YAML、包内地图和渲染底图；默认 `offline` 与训练环境一致，`scripted` 仅用于自包含调试。 |
| `evaluate_ppo_checkpoints.py` | 批量策略评测 | 加载 run 目录内 PPO checkpoint、固定 seed/profile 配置和正式地图，输出逐场/汇总 CSV 及排名。 |
| `replay_database_sentry_sparring.py` | 数据库混合诊断 | 加载外部 SQLite、`rm_rl` 数据 schema、记录哨兵轨迹和冻结陪练；它不是交互闭环的默认入口。 |
| `align_semantic_map.py` | 图像/世界坐标初始对齐 | 加载人工标注图和障碍图，输出 aligned 预览图和语义 JSON。 |
| `calibrate_semantic_map.py` | 地标标定 | 加载地标 JSON，用 affine/homography 拟合像素到比赛 `map` 坐标变换并输出残差。 |
| `validate_semantic_map.py` | 地图一致性检查 | 加载 annotation、obstacle 和 semantic JSON，检查坐标、polygon、目标点与障碍冲突并生成报告。 |
| `export_inflated_astar_map.py` | 哨兵 A* 图生成 | 加载 semantic JSON 和原始障碍 PNG，按指定净空生成确定性的预膨胀黑白图。 |
| `extract_2026_mesh_map.py` | Gazebo 网格俯视导出 | 可选加载 `trimesh` 和 RMUC DAE，输出俯视标注底图/候选占据图。 |
| `foxglove_semantic_preview.py` | ROS/Foxglove 预览节点 | 加载包内语义 JSON/PNG，发布预览 OccupancyGrid、MarkerArray 和静态 TF；不覆盖导航 `/map`。 |

## `assets/` 地图与标定资源

| 文件 | 功能/被谁加载 |
| --- | --- |
| `blackwhite_map.png` | 原始物理占据图；`SemanticMap` 用于 LOS、非哨兵通行和重新生成膨胀图。 |
| `blackwhite_astar_inflated_0p35m.png` | 按 0.35 m 净空预膨胀的哨兵中心 A* 图；正式世界优先加载。 |
| `semantic_map_aligned.json` | 28 x 15 m frame、地图文件名、区域 polygon、语义类型和标定元数据；世界与工具的主入口。 |
| `semantic_map_aligned.png` | 与世界坐标对齐的彩色场地底图；回放渲染加载。 |
| `demo_map.png` | 人工语义标注/验证底图；对齐和验证工具加载。 |
| `radar_landmark_template.json` | 雷达/地图地标标定输入模板；校准工具加载。 |
| `semantic_map_validation.json` | 最近一次地图验证统计和问题报告。 |
| `semantic_map_debug.png` | polygon、anchor、障碍叠加诊断图。 |
| `semantic_alignment_2026-07-29.md` | 当前坐标对齐来源、限制和人工检查记录。 |
| `semantic_alignment_2026-07-29_132106.png` | 对齐过程快照，供人工复核。 |
| `rmuc2026_mesh_annotation_base.png` | 从 Gazebo 网格导出的人工标注底图。 |
| `rmuc2026_mesh_occupancy_candidate.png` | 从 Gazebo 网格导出的候选占据图，不直接当作已验证通行图。 |

## `config/` 世界与训练配置

| 文件 | 功能/加载内容 |
| --- | --- |
| `arena_example.yaml` | 小型语义地图 YAML 格式示例，供 `SemanticMap.from_yaml()` 加载。 |
| `demo.yaml` | 自包含 `scripted` 世界和小规模 PPO 配置；训练 CLI 默认加载。 |
| `offline_sparring.yaml` | 单套冻结 offlineRL 陪练基线，包含各角色 run 目录。 |
| `offline_sparring_building_targets.yaml` | 带建筑 target 动作的冻结陪练配置。 |
| `online_ppo_11_offline_sparring.yaml` | 红方 PPO 对 11 个冻结陪练单位的训练配置。 |
| `online_ppo_active_hunt.yaml` | 主动选敌/追击实验配置，1 Hz 更新冻结陪练意图。 |
| `online_ppo_aggressive_sparring.yaml` | aggressive 蓝方陪练实验配置。 |
| `online_ppo_candidate_targets.yaml` | goal/target 候选动作版本的 PPO 配置。 |
| `online_ppo_hp20_inflated035.yaml` | 20% 恢复阈值和 0.35 m 哨兵膨胀图单环境配置。 |
| `online_ppo_hp20_inflated035_parallel4.yaml` | 上述规则的 4 环境并行训练配置。 |
| `replay_ppo_style_pool.yaml` | aggressive/pressure/measured 三种蓝方冻结 checkpoint 映射及回放参数。 |
| `simulation2d_live.yaml` | 实时节点完整参数参考，包括策略、checkpoint、速率、循环和所有 topic 名。 |

配置中的 `rm_runs/...` 和 `sparring_style_pool/...` 是可选外部实验资产路径；`scripted`
模式不会读取它们。

## `resource/` 与 `test/`

| 文件 | 功能 | 加载内容 |
| --- | --- | --- |
| `resource/simulation2d` | ament resource marker | 让 ROS 2 package index 找到该包。 |
| `test/test_world_model.py` | 世界闭环测试 | 加载正式地图，推进交互 step，验证 observation 尺寸和 goal/target 反馈字段。 |
| `test/test_mapping.py` | 动态代价图测试 | 构造障碍和 RadarTrack，验证 threat、A* goal 和 5 通道雷达特征。 |
| `test/test_resources.py` | 安装资源测试 | 验证关键 JSON、原始/膨胀地图和默认配置可定位。 |
| `test/test_live_engine.py` | 实时驱动测试 | 验证追击策略、逐步推进、终局停止以及自动 episode 循环复位。 |

## 边界说明

- 当前 `mapping/features.py` 的 5 通道在线雷达格式还没有自动转换成现有 PPO 的 15 通道、
  171 维 observation；后续需要单独的 live observation adapter。
- `runtime/actions.py` 只定义高层动作，不负责 Gazebo/实车的双向 ROS 通信。
- 哨兵本地导航和 ESDF 仍拥有最终安全裁决权；二维世界的 A* 不替代实车局部规划。
- `offline` 是 PPO 部署和回放的默认交互环境；`scripted` 只用于隔离外部模型依赖的调试。
