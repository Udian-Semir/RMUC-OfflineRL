# 雷达站端哨兵战术决策系统计划

更新日期：2026-08-03

本文档只描述当前有效方案、已经实现的能力和下一步工作。历史 smoke 实验、已废弃
checkpoint 和重复讨论不再保留；训练结果只作为当前 2D 世界模型内的诊断证据。

## 1. 目标与边界

唯一在线学习对象是红方哨兵。它在雷达站侧根据全局比赛状态，以 1 Hz 选择：

```text
战术目的地 goal
优先攻击目标 target
开火意图 fire_mode
```

PPO 不控制轮速、云台角度或底盘变形。实车执行仍由 Gazebo 工程中的全局规划、局部
避障、轨迹跟踪、自瞄和电控状态机完成。

其他 11 个单位不是训练目标。它们使用官方数据训练出的冻结 offlineRL 策略作为队友和
对手，为主哨兵提供会移动、交战、死亡和复活的陪练环境。

LLM 不在当前实时闭环中。当前问题是结构化地图、规则状态转移和目标选择，不需要用
文本模型代替 PPO、规则状态机或路径规划器。

## 2. 总体架构

```text
雷达站可获得的敌我位置、血量和比赛状态
                    +
黑白可通行图、语义多边形、动态威胁层
                    |
                    v
       候选 goal / target 特征与 action mask
                    |
                    v
           红方哨兵 PPO，1 Hz 决策
                    |
                    v
      goal_xy + target_id + fire_mode + TTL
                    |
                    v
哨兵本地：路径规划 -> 语义路径事件 -> 底盘模式 -> 局部避障
                    |
                    v
             云台、自瞄、发射机构
```

训练中的 `GridNavigationBackend` 只是战术级近似。实车应继续使用
`Gazebo_simulation_for_sentry` 中的 JPS/MINCO/ESDF 控制链路，雷达策略不能绕过本地
硬障碍和安全仲裁。

## 3. 当前地图与观测构造

### 3.1 地图

当前训练加载：

```text
sentry_tactical_rl/assets/semantic_map_aligned.json
sentry_tactical_rl/assets/blackwhite_map.png
sentry_tactical_rl/assets/blackwhite_astar_inflated_0p35m.png
```

战术栅格分辨率为 `0.1 m`，尺寸为 `280 x 150`，对应 `28 m x 15 m` 场地。原始
`blackwhite_map.png` 保存物理障碍；`blackwhite_astar_inflated_0p35m.png` 保存哨兵中心
可进入的位置，黑色为禁止、白色为可通行。JSON 保存两张图的关系、`0.35 m` 净空以及
基地、前哨、堡垒、补给区、中央高地、隧道增益区和起伏路等独立多边形。

主哨兵和蓝方陪练哨兵的 A* 直接使用膨胀 PNG。英雄、工程和步兵使用未膨胀物理图，
不会被哨兵尺寸限制；空中单位不使用地面障碍图。窄通道在硬净空之后保留安全中心带，
并通过障碍距离 soft cost 让 A* 选择中线；宽阔区域保留所有满足 0.35 m 净空的空间。

目前像素底图和语义多边形已经对齐。接实车前仍需确保雷达、导航和该地图使用同一个
`map` 原点、方向和米制坐标，并验证前哨固定射击点、隧道入口和补给区的实测坐标。

### 3.2 地图实现注意事项

- `0.35 m / 0.1 m` 不能简单向上取整成四格膨胀；四格等价 0.4 m，会错误封死
  `tunnel_gain_4`。当前使用可接受小数半径的欧氏膨胀，四个 tunnel 和 13 个锚点保持
  连通；
- A*、goal 候选和 action mask 使用膨胀图；射线/遮挡使用原始物理障碍图。膨胀墙是
  车体中心约束，不是真实子弹遮挡；
- 膨胀 PNG 是训练与回放的确定输入，不应每次运行临时生成不同结果。生成工具为
  `python -m sentry_tactical_rl.tools.export_inflated_astar_map`；
- 雷达侧膨胀只保证静态 goal 合理。哨兵本地仍必须用实时 ESDF 将 goal 投影到
  `clearance >= 0.35 m` 的位置，并可因动态障碍拒绝雷达 goal；
- 不同底盘使用不同通行层。禁止为了主哨兵安全而让全部 offlineRL 陪练共同使用哨兵
  膨胀图；
- 语义层、物理障碍层、哨兵 A* 层和动态威胁层必须分开保存，不能反复在同一 PNG 上
  覆盖加工。

### 3.3 PPO 观测

当前观测为：

```text
地图张量：15 x 150 x 280
结构化向量：171 维
```

171 维组成：

```text
20 个全局/自身/比赛阶段特征
13 个 goal x 5 个候选特征
6 个敌方槽位 x 11 个 target 候选特征
5 个非学习友方槽位 x 4 个实体特征
```

每个车辆 target 的 11 个特征为：

1. 相对 `dx`；
2. 相对 `dy`；
3. 目标血量比例；
4. 直线距离；
5. 目标存在标志；
6. 射击站位可达标志；
7. A* 路径 cost；
8. 路径风险；
9. 站位处预计一秒有效伤害；
10. 目标武器威胁；
11. 目标是否正在威胁己方基地。

地图通过 CNN 编码，171 维向量通过 MLP 编码，融合后分别输出 goal、target、fire 三个
离散策略头和一个 value 头。不可达 goal、死亡目标和规则禁止目标使用 action mask。

## 4. 当前 1 Hz 决策输出

### 4.1 仿真内部真实输出

当前工程没有真正发布 ROS2 topic 或 CDC 数据包。1 Hz 指的是环境每秒调用一次 PPO，
原始 action 只有三个整数：

```text
(goal_index, target_index, fire_mode)
```

目标槽位固定为：

| target_index | 当前含义 |
| ---: | --- |
| 0 | blue hero |
| 1 | blue engineer |
| 2 | blue infantry3 |
| 3 | blue infantry4 |
| 4 | blue aerial |
| 5 | blue sentry |
| 6 | blue outpost |
| 7 | blue base，当前 PPO mask 禁止 |
| 8 | none |

`fire_mode=0` 表示停火，`fire_mode=1` 表示允许在射程、视线、弹药和热量均合法时交战。

例如：

```text
(11, 3, 1)
= 选择 tunnel_gain_3 锚点
+ 锁定 blue infantry4（仿真 ID 104）
+ 允许交战
```

### 4.2 当前调度覆盖

PPO 原始 action 进入执行层前还会被规则调度：

- 哨兵血量严格低于 20%：锁定己方补给区、目标 `none`、停火；恢复到 90% 后解除；
- 60 秒后蓝前哨仍存活：强制蓝前哨目标并请求交战；
- 除强制恢复外追击车辆时：原始语义 goal 被替换成目标周围的可达射击站位；
- 最终射击站位一旦生成便锁存，在哨兵进入 `0.35 m` 到达容差后才允许普通策略切换到
  下一个 goal；target 和 fire 仍可每秒更新。回补给和 60 秒前哨命令可抢占，A* 不可达
  时释放锁存以避免死锁；
- 所有动态 goal 先投影到哨兵膨胀图，再通过 A* 可达性检查；实车导航收到后还会用本地
  ESDF 做最终投影。

### 4.3 部署接口现状

`sentry_tactical_rl/deployment.py` 目前只是 JSON helper，不是可直接上车的桥接节点。已知
缺口：

- 没有 ROS2 publisher 或 CDC encoder；
- 没有把仿真内部动态计算的 `execution_goal_cell` 输出；
- 仍残留旧 1 m cell 坐标假设，不能直接用于当前 0.1 m 地图；
- 当前仿真 roster 的第 4 槽是 aerial，Gazebo 协议对应槽位仍写成 infantry5；
- 没有底盘变形字段和执行反馈。

正式 1 Hz 下发建议统一为：

```text
sequence / timestamp / ttl
action_type
goal_x_m / goal_y_m
target_type / target_roster_index
fire_mode
desired_standoff_m
route_profile
constraint_flags
fallback_action
```

追击 blue infantry4 的逻辑示例：

```json
{
  "type": "PURSUE",
  "target": {"type": "ENEMY_ROBOT", "roster_index": 3},
  "goal_map_m": {"x": 14.2, "y": 7.6},
  "fire_mode": "ENGAGE_IF_LEGAL",
  "desired_standoff_m": 2.5,
  "ttl_ms": 800,
  "route_profile": "SAFE",
  "fallback": "LOCAL_UTILITY"
}
```

坐标只用于说明 schema，真实 goal 必须来自当秒目标位置、A* 和射击站位计算。

### 4.4 隧道变形职责

PPO 不应按 1 Hz 直接控制变形时机。推荐分工：

```text
雷达站：下发战术 goal、target、期望距离和 route profile
哨兵本地：对规划路径做语义前视
          -> 即将进入 tunnel polygon 时请求变形
          -> 收到底盘完成 ACK 后允许进入
          -> 离开多边形并经过滞回距离后恢复
```

这样变形由高频定位、实际路径和底盘反馈决定，不依赖雷达通信恰好在入口处到达。若需要
雷达提示，可以增加 `route_contains_tunnel`，但它只能作为预告，不能代替本地状态机。

## 5. 2D 世界模型

### 5.1 比赛规则

当前确定性规则状态机覆盖：

- 单局 420 秒；
- 基地 `5000 HP + 150` 虚拟护盾；
- 前哨 `1500 HP`，前哨存活时基地无敌；
- 基地每累计损失 1000 实际 HP 获得一次前哨重建机会；
- 300 秒后关闭前哨重建；
- 普通单位占领 10 秒、工程占领 5 秒后重建 750 HP 前哨；
- 地面单位死亡后回己方补给区复活，并获得 30 秒无敌；
- 补给区通常按最大 HP 的 10%/s 回血；240 秒后且脱战 6 秒时为 25%/s；
- 官方终局和同分比较顺序。

尚未完整实现金币立即复活、完整等级/经济、精确弹道、姿态命中盒和全部裁判交互。

### 5.2 战斗模型

双方距离伤害对称：

| 攻击者 | 0--3 m | 3--5 m | 其他 |
| --- | ---: | ---: | ---: |
| 普通有效 DPS 单位 | 140 HP/s | 70 HP/s | 0 |
| 哨兵 | 180 HP/s | 100 HP/s | 0 |

英雄保持 4 秒一次的 200 HP 单发节奏。建筑伤害还需满足建筑可攻击、视线、射程、开火
意图、附近无优先车辆目标和连续停留等规则。空中单位可以攻击，但不接受地面单位伤害。

### 5.3 陪练

红方主哨兵之外的 11 个角色使用冻结 IQL/offlineRL checkpoint。offlineRL 负责高层移动
goal 和目标意图；2D 世界负责 A*、视线、距离、弹药、热量和伤害结算。当前默认
`sparring_fire_policy=intent_dps`，因为官方稀疏 `fire_gate` 在构造的 2D 状态中存在明显
分布偏移，不能直接当成低层开火物理条件。

`aggressive / pressure / measured` 已按整局/阵营的官方日志统计构建为互斥的数据集：
`pressure` 偏敌建筑伤害和前压，`aggressive` 偏交火密度和前压，`measured` 为其余较稳健
局面。`sparring_style_pool/` 中已经完成 3 个风格 x 5 个角色的 15 个 IQL checkpoint；
步兵 3/4 共享同一个 infantry checkpoint。环境的
`blue_sparring_style_run_dirs` 会在每局开始选定一个蓝方风格，并为蓝方各角色加载该风格的
冻结策略，而不是只改变开火概率。`replay_ppo_style_pool.yaml` 保存了完整映射，并有回归测试
验证不同风格实际选择不同的 checkpoint。

当前 `intent_dps` 火控模式仍会绕过旧的 auto-aim 概率，因为它只把 offlineRL 输出视为
高层 goal/target 意图，再由 2D 规则结算合法伤害。因此多风格的差异来自 checkpoint 的
移动和选敌，不应从 `blue_autoaim_probability` 判断。下一次红方 PPO 正式训练应显式使用这
套 style-pool 配置，并以不同风格、不同随机种子分别评测；该 PPO 对照实验尚未因此文档而
自动完成。

## 6. 当前策略和 reward

### 6.1 现行策略

- 60 秒前或蓝前哨已毁：PPO 自主选择车辆、语义 goal 和开火；
- 60 秒后蓝前哨仍存活：规则层强制攻击蓝前哨；
- 血量严格低于 20%：回补给区；
- 其他血量下追击车辆：始终生成约 2.5 m 射击站位；
- 己方前哨已毁且敌方威胁基地：目标候选增加 `base_pressure`，打击基地攻击者获得额外
  reward。

### 6.2 当前 reward 权重

主要单步项：

| 项 | 当前值 |
| --- | ---: |
| 时间 | `-0.01 / s` |
| 无效 goal | `-0.20` |
| goal 切换 | `-0.02` |
| 导航 cost | `-0.001 * min(cost, 250)` |
| 哨兵车辆伤害 | `+0.035 * damage` |
| 哨兵前哨伤害 | `+0.030 * damage` |
| 哨兵受到伤害 | `-0.030 * damage` |
| 己方前哨受到伤害 | `-0.012 * damage` |
| 己方基地受到伤害 | `-0.040 * damage` |
| 打击基地攻击者 | 额外 `+0.050 * damage` |
| 哨兵死亡 | `-8` |
| red win / blue win | `+12 / -10` |

非恢复状态下选择 `none`、停火或不接近目标还会受到搜索/追击塑形。冻结队友造成的正向
伤害只写世界遥测，不给主哨兵 PPO 记正 credit。

## 7. 已修复问题：156 HP 隧道龟缩

标准回放 `seed=142` 在 `t=229 s` 的状态为：

```text
red sentry: 156 / 400 HP，alive=true
position: tunnel_gain_3
goal: tunnel_gain_3
target: blue infantry3
fire_mode: engage
```

它从约 216 秒一直以 156 HP 存活并停在该位置，331 秒才死亡。截图中的停留不是复活
读条，而是真实策略漏洞。

根因是旧版本把恢复和“健康追击”设成两个不连续的血量区间，中间血量既不回补给，
也不启用车辆追击站位和 pursuit reward。

156/400=39%，因此 target 虽然锁定 blue infantry3，执行层却没有把 tunnel 锚点替换成
目标周围的 2.5 m 射击位；哨兵已经位于所选锚点，A* 返回原地，目标又不在有效射程，
最终形成“存活、允许开火、但不移动也打不到”的状态。

当前代码已经删除额外的高血量门槛，并把恢复触发改为严格低于 20%。只要没有进入恢复状态，
车辆 target 都会覆盖无关语义锚点并生成 2.5 m 射击站位。旧 `ppo_00090.pt` 仍是在旧
执行分布下训练出的诊断 checkpoint，不能因为代码修复就直接视为可部署策略。

## 8. 下一版确定策略

### 8.1 车辆交战距离

- 哨兵锁定地面车辆且未进入强制恢复时，始终生成目标周围的可达射击站位；
- 期望距离固定为 `2.5 m`，合格带为 `2.0--3.0 m`；
- 进入 2--3 m 后不再奖励继续靠近，过近应重新选环上站位；
- 站位联合考虑 A* cost、动态威胁、视线和退路，不按欧氏距离直接贴脸；
- target 存活但连续数秒无距离进展、无伤害且原地不动时，触发重新规划或重新选敌。

### 8.2 前哨攻击点

攻击蓝前哨不再选择“离前哨最近的任意语义锚点”。使用一个经过校准的固定射击点：

```text
位于蓝前哨朝中央高地方向
距蓝前哨中心约 4.0 m
必须可通行、对前哨有视线、允许原地稳定开火
```

初始几何候选可按下式生成，再投影到最近的合法射击点：

```text
p_attack = p_blue_outpost
         + 4.0 * normalize(p_central_highland - p_blue_outpost)
```

正式训练前应把最终点作为明确 JSON 战术锚点保存，例如
`blue_outpost_attack_from_highland`，不要每次运行临时猜坐标。红蓝训练或部署时按阵营镜像。

### 8.3 隧道语义

- tunnel 默认是路径上的过渡语义，不是锁定车辆时的最终驻留点；
- 有车辆 target 时，最终 goal 必须是车辆射击站位，tunnel 只能出现在 A* 路径中；
- 只有明确的穿越、增益触发或防守 skill 才允许 tunnel 作为最终 goal；
- 增加 `stationary_in_tunnel` 遥测和超时回归测试；
- 实车底盘变形由本地路径语义状态机处理。

### 8.4 血量调度

血量调度只保留两个执行状态，不再另设高血量进攻门槛：

```text
HP < 20%：强制回补给，目标 none，停火；恢复到 90% 后解除
HP >= 20%：继续主动追击，并控制在 2.5 m 左右交战
```

20% 是严格触发线，正好 20% 不回补给。已经进入恢复状态后保持到 90%，用于避免补给区
边缘反复切换。执行层保证所有非恢复状态下选中敌人后不会停在无射界的 tunnel 锚点。

## 9. 实施顺序

### P0：先修执行与接口

1. [已完成第一版] 蓝前哨朝中央高地方向投影约 4 m 的固定合法射击位；正式部署仍需用实测坐标替换；
2. [已完成] 删除额外高血量门槛，车辆 2.5 m 追击站位除强制恢复外始终生效；
3. [已完成] 输出 0.35 m 欧氏膨胀 A* PNG，并按哨兵/非哨兵拆分通行层；
4. 增加无进展/无伤害 watchdog，禁止存活哨兵长期停在无射界 goal；
5. 将 tunnel 从“车辆锁定时的最终 goal”降为路径过渡语义；
6. 修正 `deployment.py` 的 0.1 m 坐标、动态 execution goal 和 roster；
7. 明确定义 1 Hz ROS/CDC bridge schema；
8. 在哨兵本地导航实现 tunnel 路径前视、变形 ACK 和退出滞回。

当前执行层还增加了一条前哨攻击保护：60 s 后若当前一秒内所有可合法命中的敌方单位
合计足以击穿哨兵剩余 HP，则中断前哨目标，优先锁定单秒伤害最大的威胁单位；威胁不再
致死后恢复前哨固定点。该规则已有回归测试，不能替代未来更长时域的 TTC/预测威胁模型。

### P1：规则回归后再训练

1. 增加 2--3 m 距离带 reward，不奖励无条件越靠越近；
2. 增加前哨固定射击点到达、保持和有效输出 reward；
3. 增加中血量、基地受压、目标死亡、路径穿 tunnel 等 curriculum 场景；
4. [已完成陪练池] 使用官方数据训练并校验 15 个多风格 IQL checkpoint；下一步用这些
   checkpoint 和多个 seed 重训、评测红方 PPO；
5. 可从 `ppo_00090.pt` 做兼容 warm start，但必须同时保留新初始化对照；
6. 不继续把 `ppo_00120.pt` 当作默认续训起点，因为当前执行分布和 reward 即将改变。

### P2：接 Gazebo 和实车

1. 统一地图坐标、roster 和战术 action 协议；
2. 将雷达动态 target 转成米制射击 goal；
3. 记录本地规划接受/拒绝、底盘模式、变形完成、到点和交战反馈；
4. 通信超时后按 `RADAR_ACTION -> RADAR_STATE -> LOCAL_ONLY` 降级；
5. 实测校准移动耗时、变形提前量、射击距离和定位误差，再回灌 2D 世界。

## 10. 必须通过的测试

### 场景回归

- 156/400 HP、位于 `tunnel_gain_3`、锁定 blue infantry3：必须生成车辆射击站位并离开
  原地，不能切换 recovery 或持续空转；
- 锁定任意地面车辆：最终站位保持在 2.0--3.0 m，且路径不穿硬障碍；
- 60 秒后蓝前哨存活：goal 固定为高地侧 4 m 射击点，不进入前哨中心；
- 前哨射击点不可达或失去视线：动作被拒绝并返回替代候选；
- 目标死亡/丢失：target 和 goal 在 TTL 内重选；
- 路径经过 tunnel：雷达 action 保持追击目标，本地状态机产生一次进入和一次退出事件；
- 低血恢复、复活、基地保护和前哨重建规则继续通过回归。

### 策略验收

- 至少多个未见 seed x 三种陪练 profile；
- 分开统计哨兵车辆伤害、哨兵前哨伤害、受到伤害和全队建筑伤害；
- 统计 2--3 m 交战占比、无进展时长、tunnel 驻留时长和无效动作；
- 对比固定守家、规则 utility、旧 `ppo_00090` 和新 PPO；
- 单场好看视频不能代替批量胜率、建筑存活和行为约束结果。

## 11. 当前训练与回放基线

当前候选目标配置：

```text
sentry_tactical_rl/configs/online_ppo_candidate_targets.yaml
```

续训目录：

```text
runs/sentry_ppo_candidate_targets_20260803_continued/
```

固定 seed `1001`、三 profile 的小样本盲测中：

| checkpoint | cases | win rate | sentry objective damage | return |
| --- | ---: | ---: | ---: | ---: |
| `ppo_00090.pt` | 3 | 1.000 | 3080 | 55.11 |
| `ppo_00100.pt` | 3 | 1.000 | 3080 | 54.78 |
| `ppo_00105.pt` | 3 | 1.000 | 1590 | -39.41 |

该评估只有一个 seed，且三个 profile 在该局得到相同统计，不能证明泛化。

诊断视频：

```text
runs/sentry_ppo_candidate_targets_20260803_continued/
  replay_seed142_aggressive_ppo00090/replay.mp4
```

该局完整运行 420 秒并为 `red_win`，但同时暴露了 156 HP 哨兵在 tunnel 存活停留超过
100 秒的问题。因此下一步不是盲目增加 PPO update，而是先完成 P0 的执行约束和接口修正。

### 20% 回补与 0.35 m A* 图微调记录

修复执行逻辑后，使用 `ppo_00090.pt` 只加载模型/value 权重，重置 Adam，并以 `1e-4`
学习率追加 60 次 PPO 更新：

```text
config: sentry_tactical_rl/configs/online_ppo_hp20_inflated035.yaml
run: runs/sentry_ppo_hp20_inflated035_v1/
updates: 91--150
```

训练后半段回报和熵出现退化，因此没有直接选最后的 `ppo_00150`。固定 seed `1001`、
aggressive 陪练下，`ppo_00100 / 00105 / 00110` 的确定性动作和结果完全一致：

| checkpoint | outcome | sentry objective damage | damage taken | return |
| --- | --- | ---: | ---: | ---: |
| `ppo_00100.pt` | blue_win | 1180 | 1880 | -92.73 |
| `ppo_00105.pt` | blue_win | 1180 | 1880 | -92.73 |
| `ppo_00110.pt` | blue_win | 1180 | 1880 | -92.73 |

使用最早的代表 checkpoint `ppo_00100.pt` 生成 seed `142` 标准回放：

```text
runs/sentry_ppo_hp20_inflated035_v1/
  replay_seed142_aggressive_ppo00100/replay.mp4
```

该局完整运行 420 秒，哨兵造成车辆 580 HP、蓝前哨 300 HP，累计受到 1880 HP，最终
`blue_win`。正向结果是：存活哨兵在 tunnel 同一位置最长只停 1 秒，旧版 156 HP 长时间
龟缩已消失。负向结果是：当前 reward 和候选站位还没有把持续追击转化为足够的交战收益，
策略强度明显低于旧诊断局。该 checkpoint 只用于验证规则修复，不作为部署候选；下一轮
应先完成前哨高地侧固定 4 m 射击点、2--3 m 距离带 reward 和无进展 watchdog，再训练。

### 4 环境并行 PPO 与逐秒发布指令回放

并行采样实现位于 `sentry_tactical_rl/parallel_env.py`。4 个 Linux fork worker 继承同一份冻结
offlineRL 权重页，各自使用间隔 seed 运行独立比赛；15 通道栅格观测写入共享内存，Pipe 只传
动作、reward、done 和小型 info。PPO 每个环境采样 32 步，一次更新仍为 128 样本，因此不会
把 rollout 内存直接扩大四倍。GAE 的 bootstrap 和 done mask 保留 `[time, env]` 两个轴，最后
才展平为训练 batch，避免把一局终点接到另一局轨迹上。

```text
config: sentry_tactical_rl/configs/online_ppo_hp20_inflated035_parallel4.yaml
run: runs/sentry_ppo_hp20_inflated035_parallel4_v1/
warm start: runs/sentry_ppo_hp20_inflated035_v2/ppo_00130.pt
updates: 131--190
workers: 4
samples/update: 32 x 4 = 128
wall time: 1005 s
peak main-process RSS: 1.27 GB
```

固定 `seed=1001`、aggressive 陪练的 12-checkpoint 筛选中，`ppo_00135` 和 `ppo_00140`
保持 warm-start 行为并获胜，之后多数 checkpoint 退化。最佳 `ppo_00135` 的结果为：车辆
伤害 1440 HP、蓝方前哨伤害 940 HP、累计承伤 1880 HP，红方前哨和基地均未掉血，最终
`red_win`，return 为 `-19.51`。因此并行化已验证速度和样本独立性，但这轮没有证明策略
强于 warm-start；下一轮应降低策略漂移并增加多 seed/profile 选模，不能继续把最后模型当最佳。

新版回放逐秒记录并显示规则层最终执行指令：地图用橙色十字标出发布的 `goal(x,y)`，顶部
打印 `target` 和 `fire`。420 行 `steps.csv` 均包含 `published_goal_x_m`、
`published_goal_y_m`、`published_target` 和 `published_fire`。代表视频为：

```text
runs/sentry_ppo_hp20_inflated035_parallel4_v1/
  replay_seed1001_aggressive_ppo00135_goals/replay.mp4
```

### 固定前哨点与致死威胁优先训练

在上述并行训练器上应用前哨固定点和致死威胁打断规则后，从 `parallel4_v1/ppo_00135.pt`
重置优化器续训 60 更新：

```text
config: sentry_tactical_rl/configs/online_ppo_hp20_inflated035_parallel4.yaml
run: runs/sentry_ppo_hp20_inflated035_parallel4_v2/
updates: 136--195
```

固定 `seed=1001/aggressive` 评估中，最佳为 `ppo_00155.pt`：红方胜，哨兵车辆伤害
1600 HP、哨兵对蓝前哨伤害 920 HP、累计承伤 1835 HP，红方前哨和基地均未掉血，return
`-2.00`。回放中 420 个 1 Hz 指令全部记录了发布 goal/target；其中 228 秒使用固定
`goal=(16.35, 6.65)m`，在 `t=204/270/408 s` 发生致死威胁打断，临时目标分别为
`blue infantry4 #104`、`blue infantry3 #103`、`blue infantry3 #103`。

该 4 m 规则视频已被后续 7 m/aerial 修正版替代，旧媒体文件按磁盘清理要求删除。

### Aerial 建筑攻击执行修正

此前回放中 aerial 虽然多次输出前哨 target，但因为一直沿用离线导航 goal，且空中单位的
`fire_allowed` 全程为 false，最终只在偶然经过射程时攻击。现在执行器只在 aerial 明确选中
建筑时耦合到稳定的空中攻击位；空中射击不使用地面黑白图的 LOS，车辆和地面单位仍使用
普通射线。使用现有 `ppo_00155` 做不训练快速回放验证，blue aerial 对红方前哨产生 8 次
结构攻击、累计 1500 HP，red aerial 对蓝方前哨同样产生 8 次、累计 1500 HP。该修正改变
了陪练动力学，正式 PPO 结果需要在此执行器上重新训练和评估。

当前代表回放：

```text
runs/sentry_ppo_hp20_inflated035_parallel4_v2/
  replay_seed1001_aggressive_ppo00155_aerialfix_goals/replay.mp4
```

该局双方 aerial 都在 60 s 前摧毁对方前哨，因此哨兵的 7 m 固定前哨 goal 没有触发。
下一轮训练前需要确认 aerial 连续 `200 HP/s` 是否符合期望的命中率和资源约束；这一局
只能验证 aerial 执行器，不能验证哨兵 7 m 前哨攻击点。

## 12. 当前需要的外部校准

后续接入需要明确以下实车参数：

1. 蓝/红前哨高地侧固定射击点的最终 `map(x, y)`；
2. tunnel 入口/出口多边形与底盘变形需要的提前距离；
3. 变形请求、完成、失败和恢复的电控状态接口；
4. 真实导航对战术 goal 的接受/拒绝反馈；
5. 车辆 2--3 m 交战时的视线、命中率和底盘机动边界；
6. 雷达 roster 与裁判/哨兵端 target ID 的唯一映射。

这些参数进入配置和回归测试后，才开始下一轮正式 PPO 训练。
