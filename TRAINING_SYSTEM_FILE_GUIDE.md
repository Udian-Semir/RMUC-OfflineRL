# 训练系统文件索引与迁移规则

本文对应当前仓库的有效训练链路。它区分三件事：可复用的工具代码、某张地图上的数据，以及
在这张地图和这套规则上学到的 checkpoint。最后两者不能因为模型文件能加载就被误认为可直接
迁移。

## 换地图或规则时能否使用

可以继续使用同一套工具，但要按改变的范围处理：

| 改动 | 可直接复用 | 必须重新验证/训练 |
| --- | --- | --- |
| 只替换底图、障碍和语义区 | `SemanticMap`、膨胀图导出、校准、A*、代价图、PPO/回放工具 | 重新导出膨胀图、校准/连通性测试、重新训练 PPO；有真实新地图日志时重建 offlineRL 陪练数据并重训 |
| 小幅调整伤害、回血、时间等数值，观测和动作不变 | 数据格式、网络结构、训练器 | 规则单元测试、基准回放、PPO 重新训练和多 seed 评测；旧 PPO 仅可作为 warm start 对照 |
| 新增建筑/兵种/增益、修改胜负规则，或改变观测/动作含义 | 通用工具和可视化框架 | 修改状态转移和测试，重建数据标签，重训 offlineRL 陪练与 PPO；旧 checkpoint 不可作为结果 |

原因是 offlineRL 学到的是“旧日志在旧坐标和规则下的高层行为”，PPO 学到的是“当前环境的
转移和 reward”。它们不是与地图无关的抽象规则库。地图改变后即使张量尺寸相同，旧空间位置的
价值也已变；规则改变后，同一个动作的后果也已变。

新地图/规则的最小执行顺序是：

1. 更新 `assets/blackwhite_map.png`、`assets/semantic_map_aligned.json` 和规则参数；不要把语义区画进物理障碍 PNG。
2. 用 `tools/export_inflated_astar_map.py` 生成哨兵专属通行图，再运行地图校验、地标校准和 A* 连通性测试。
3. 修改 `match_rules.py` 与 `env.py` 的状态转移，给每条新规则增加场景测试。
4. 有同规则、同地图的比赛日志时，用 `rm_rl/data/build_dataset.py` 重建陪练数据并训练风格池；没有日志时，不能把旧地图的 offlineRL 叫作新地图的真实模仿策略。
5. 以冻结陪练池运行 PPO，保留新初始化和兼容 warm start 两组，多 seed 评测后再选 checkpoint。

## 顶层资料与产物

| 文件或目录 | 职责 |
| --- | --- |
| `Sentry_radar_side_decision.md` | 当前雷达侧单红方哨兵方案、地图、规则、reward、已知边界和实施顺序。 |
| `README.md` | 原始 OfflineRL 项目的快速入口。 |
| `configs/*.yaml` | offlineRL 的数据路径、算法和训练超参数；`*_iql_tactical.yaml` 是当前陪练训练主配置。 |
| `dataset/rmuc_2026_region_dataset.sqlite` | 官方裁判日志源数据，不应被训练程序改写。 |
| `data/` | 从 SQLite 导出的角色数据集、经验交战图和队伍先验。地图/规则/特征定义改变后需要检查或重建。 |
| `rm_runs/` | 基础角色 offlineRL 输出。每个运行目录的 `best.pt`、`meta.json`、`norm_stats.npz` 和 CSV 必须成套使用。 |
| `sparring_style_pool/` | 三种风格的可审计数据、15 个 checkpoint、角色报告和风格分配清单。 |
| `runs/` | 红方 PPO checkpoint、训练指标、回放 CSV/视频和固定评测结果；不同地图或规则不得混用。 |

## OfflineRL 数据与陪练：`rm_rl/`

| 文件 | 职责 |
| --- | --- |
| `rm_rl/data/schema.py` | 裁判 SQLite 的表、兵种、阵营、HP 等字段常量及角色解析，是日志契约。 |
| `rm_rl/data/features.py` | 将某一秒裁判状态编码成固定 161 维观测，并定义战术目标标签顺序。`sparring_adapter.py` 复用它，不能私自改变顺序。 |
| `rm_rl/data/build_dataset.py` | 从 SQLite 按整局切分 train/validation，构造状态、下一步战术动作、reward 和归一化统计。 |
| `rm_rl/data/build_sparring_style_pool.py` | 按整局/阵营的前压、交火、建筑伤害统计分为 aggressive/pressure/measured，并为每个角色写出数据集。 |
| `rm_rl/data/dataset.py` | 读取 NPZ transition 数据、batch 和归一化统计。 |
| `rm_rl/data/check_dataset.py` | 检查 NaN、维度、时间范围、动作和正样本比例；训练前必须运行。 |
| `rm_rl/data/reward.py` | 从日志派生离线训练使用的即时 reward。 |
| `rm_rl/data/reward_model.py` | 训练/调用可选 reward model 的辅助逻辑，不是 2D 世界的规则状态机。 |
| `rm_rl/data/team_prior.py` | 从历史比赛建立无泄漏队伍先验；合成世界没有学校身份时 adapter 使用中性先验。 |
| `rm_rl/data/vis_map.py` | 从日志统计经验交战/可见性地图，供数据特征使用，不取代黑白物理障碍图。 |
| `rm_rl/algos/action_spec.py` | 规定速度或 tactical 动作的维度、打包和反归一化。动作语义变化时先改这里。 |
| `rm_rl/algos/networks.py` | BC/IQL/DT 共用的 MLP 等神经网络组件。 |
| `rm_rl/algos/iql.py` | IQL 的 Q、V、actor 损失和更新；只从历史 transition 学习，不在世界中探索。 |
| `rm_rl/algos/bc.py` | 行为克隆基线。 |
| `rm_rl/algos/dt.py` | Decision Transformer 模型。 |
| `rm_rl/train/common.py` | 配置、随机种子、CSV 日志、checkpoint 等共用训练工具。 |
| `rm_rl/train/train_offline.py` | BC/IQL 的 CLI 训练入口，保存 `best.pt`/`final.pt` 和日志。 |
| `rm_rl/train/train_dt.py` | DT 的训练入口。 |
| `rm_rl/deploy.py` | `MLPPolicyRunner`：读取 checkpoint、归一化观测、反归一化动作。环境中的冻结陪练通过它推理。 |
| `rm_rl/eval/ope_fqe.py` | FQE 离线策略评估，只能估计日志分布内价值，不能证明仿真或赛场胜率。 |
| `rm_rl/eval/ope_dt.py` | DT 的离线评估。 |
| `rm_rl/eval/write_role_report.py` | 生成每角色报告，解释 `action_mse`、`fire_f1`、`target_top1_named` 等。 |
| `rm_rl/eval/render_offline_replay.py` | 把离线动作/轨迹绘制为回放。 |
| `rm_rl/eval/plot_sparring_scenarios.py` | 绘制陪练场景诊断图。 |
| `rm_rl/eval/show_offline_points.py` | 查看离线策略在指定局面的目标点/动作。 |
| `rm_rl/eval/win_alignment.py` | 检查策略/数据指标与比赛终局的关系。 |

## OfflineRL 运行脚本：`scripts/`

| 文件 | 职责 |
| --- | --- |
| `01_build_sentry_dataset.sh` | 构建旧的单哨兵离线数据流程。 |
| `02_train_sentry_iql.sh` | 训练旧的单哨兵 IQL 基线。 |
| `03_train_sentry_dt.sh` | 训练旧的单哨兵 DT 基线。 |
| `train_sparring_roles.sh` | 构建、检查、训练 hero/engineer/aerial/blue-sentry；步兵沿用主流程。 |
| `train_sparring_style_pool.sh` | 依次检查并训练 3 风格 x 5 角色，写 `ROLE_REPORT.md`。 |
| `run_all.sh` | 原始端到端实验脚本：交战图、队伍先验、数据、IQL/BC/DT、OPE、导出。 |
| `collect_results.sh` | 收集多次实验结果。 |
| `export_policy.py` | 导出 checkpoint 以供部署侧加载。 |

## 2D 世界、雷达特征与红方 PPO：`sentry_tactical_rl/`

| 文件 | 职责 |
| --- | --- |
| `semantic_map.py` | 读取 JSON/YAML 地图、像素与米坐标转换、语义区、多层障碍、A*、视线和栅格通道。换图的核心入口。 |
| `radar_costmap.py` | ROS 静态图加膨胀、敌方 track 威胁场和 A* 代价；属于传统安全/路径层，不由 RL 学习。 |
| `radar_features.py` | 将候选语义锚点、路径 cost/risk、动态威胁转为 PPO 特征。 |
| `navigation.py` | `GridNavigationBackend` 的训练期近似导航；实车应由 Gazebo/本地规划器接管。 |
| `match_rules.py` | 420 秒、基地/前哨 HP、护盾、重建机会、终局等确定性比赛状态转移。换规则先改这里并新增测试。 |
| `env.py` | 2D 世界主环境：单位 roster、回合推进、伤害/复活/回血、语义效果、reward、动作 mask、红方哨兵调度和冻结陪练执行。唯一在线学习主体是红方哨兵。 |
| `sparring_adapter.py` | 将环境快照严格转成 offlineRL 的 161 维状态，调用冻结策略，并把输出变为高层 goal/target/fire intent。 |
| `model.py` | CNN 地图编码器 + MLP 向量编码器，输出 goal/target/fire 三个策略头和 value。 |
| `ppo.py` | PPO rollout、GAE、裁剪目标、更新、存取 checkpoint；不枚举所有策略。 |
| `parallel_env.py` | Linux fork 的多局并行采样与共享观测缓冲；加速的是采样吞吐，不改变单局 420 秒的模拟时间尺度。 |
| `train.py` | 红方 PPO CLI：读 YAML、选 CPU/CUDA、支持续训、CSV/dashboard、checkpoint。 |
| `live_plot.py` | 保存 `training_metrics.csv` 并可显示实时 reward、cost、伤害等曲线。 |
| `deployment.py` | 当前只是战术 JSON helper，尚不是 ROS2/CDC 上车节点；其中旧 1m cell 假设仍需修正后才能接实车。 |
| `foxglove_preview.py` | 发布地图/语义区用于 Foxglove 对齐检查，不可代替雷达坐标标定。 |
| `costmap_smoke.py` | 用 ROS map YAML 快速检查传统代价图链路。 |
| `smoke.py` | 不训练的最小环境契约检查。 |
| `__init__.py` | Python 包标记，无训练逻辑。 |

## 地图、配置和工具

| 文件或目录 | 职责 |
| --- | --- |
| `assets/blackwhite_map.png` | 原始静态物理障碍和射线遮挡层。 |
| `assets/blackwhite_astar_inflated_0p35m.png` | 0.35m 哨兵中心通行层，只用于哨兵 goal/A*，由导出脚本确定性生成。 |
| `assets/semantic_map_aligned.json` | 米制边界、像素映射、语义多边形、锚点和各兵种通行层的声明，是换图时最重要的标注文件。 |
| `assets/semantic_map_aligned.png` / `semantic_map_debug.png` | 人工检查用的渲染图，不作为物理真值。 |
| `assets/semantic_map_validation.json` | 地图验证输出。 |
| `assets/radar_landmark_template.json` | 填入真实地标以求雷达 map 与语义图的变换。 |
| `configs/demo.yaml` | 小型冒烟 PPO 配置。 |
| `configs/arena_example.yaml` | 外部语义地图配置格式示例。 |
| `configs/offline_sparring*.yaml` | 使用基础冻结陪练的环境配置。 |
| `configs/online_ppo_*.yaml` | 不同阶段的红方 PPO 实验配置；输出目录应保持唯一。 |
| `configs/online_ppo_hp20_inflated035_parallel4.yaml` | 420 秒、0.35m 膨胀地图、4 并行环境的当前基线配置。 |
| `configs/replay_ppo_style_pool.yaml` | 回放时把蓝方映射至三个真实风格 checkpoint 的完整配置。 |
| `tools/export_inflated_astar_map.py` | 从原始障碍图生成固定净空图。 |
| `tools/validate_semantic_map.py` | 检查 JSON、区域、锚点、净空和连通性。 |
| `tools/calibrate_semantic_map.py` | 用地标拟合像素到真实 `map` 坐标的 affine/homography。 |
| `tools/align_semantic_map.py` | 辅助对齐语义标注与底图。 |
| `tools/extract_2026_mesh_map.py` | 从 Gazebo RMUC 2026 网格导出人工标注底图，不能直接当可通行图。 |
| `tools/foxglove_semantic_preview.py` 和 `run_foxglove_semantic_preview.sh` | 发布 Foxglove 预览和启动包装。 |
| `tools/replay_offline_sparring.py` | 载入 PPO + 冻结陪练，连续运行一整局、输出逐秒 CSV、渲染帧/视频。 |
| `tools/replay_renderer.py` | 回放的地图、单位、HP、目标和事件渲染器。 |
| `tools/replay_database_sentry_sparring.py` | 用官方记录的红哨兵状态驱动混合回放。 |
| `tools/evaluate_ppo_checkpoints.py` | 在固定 seed/陪练条件下横向比较多个 PPO checkpoint。 |

## 测试与如何判断可用

| 文件 | 覆盖的契约 |
| --- | --- |
| `tests/test_calibrate_semantic_map.py` | 坐标拟合和残差。 |
| `tests/test_foxglove_preview.py` | 地图和语义区预览输出。 |
| `tests/test_match_rules.py` | 前哨保护、基地护盾、重建、终局规则。 |
| `tests/test_tactical_building_targets.py` | offlineRL 建筑目标标签与伤害事件是否同一 transition。 |
| `tests/test_tactical_semantics.py` | 地图净空/连通、陪练 adapter、风格切换、伤害、复活、追击、前哨和基地语义。 |
| `tests/test_parallel_ppo.py` | 并行 rollout 维度、done 边界和最终下发 action 遥测。 |

在当前工作树，`python -m unittest discover -s tests -v` 已通过 47 项。它证明代码契约没有
明显回归，不等于策略已经足以比赛。策略验收还必须看：固定 seed 下的全场回放、红方哨兵实际
伤害/承伤/前哨与基地结果、路径是否合法、不同陪练风格和 seed 下的均值与方差，以及实车雷达/
导航闭环的坐标和时延误差。

## Checkpoint 与日志阅读

`best.pt` 是验证集指标最好的 offlineRL 权重；`final.pt` 是最后一步权重；`ckpt_*.pt` 是中间
恢复点。`meta.json` 记录观测/动作定义，`norm_stats.npz` 记录标准化参数，两者必须与权重配套。
PPO 的 `ppo_*.pt` 同时保存策略、value 和优化器状态；仅在地图、观测、动作和规则相容时才可
用 `--resume` 继续。跨规则 warm start 应使用 `--reset-optimizer`，并且必须与新初始化实验比较。

离线角色首先看同一角色、同一数据版本下的 `action_mse`、`fire_f1`、`target_top1_named` 和
回放；不要把它们解释为胜率。PPO 首先看多 seed 完整对局的终局、实际伤害与建筑状态、reward
分项和行为视频；单条 reward 曲线只说明优化过程是否稳定，不能单独证明战术更强。
