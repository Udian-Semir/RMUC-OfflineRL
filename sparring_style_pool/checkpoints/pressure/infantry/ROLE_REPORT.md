# pressure_infantry OfflineRL Report

## Dataset

- agent types: `步兵3, 步兵4`
- action mode: `tactical`
- observation/action dimensions: `161 / 12`
- train: `734` episodes, `305802` transitions
- validation: `74` episodes, `30646` transitions
- match duration: `420.0 s`
- dataset sanity check: `not recorded`

## Data Semantics

- Red and blue trajectories are both included; blue trajectories are rotated into the red-centric ego frame before training.
- The validation split is by match, not randomly sampled seconds from the same match.
- Tactical navigation is the normalized 5-second ego displacement. The fire gate comes from the selected weapon's ammunition counter. Target labels use turret/HP evidence where available; because the referee log omits target IDs, a weak building-intent label is also added after >2 s stationary in 5 m structure range with no enemy vehicle in range.

## Selected Checkpoint

- use `best.pt` for deployment/evaluation

## Best Validation Row

- step: `2000`
- val_loss: `4.183385252952576`
- action_mse: `0.06860059723258019`
- nav_mse: `0.06927634105086326`
- fire_acc: `0.8164202908674876`
- fire_f1: `0.22388292752827207`
- fire_pos_rate: `0.1797303907573223`
- target_top1: `0.49972204168637596`
- target_top1_named: `0.14605280458927156`
- target_named_rate: `0.48160218000411986`
- building_target_rate: `0.0205527707003057`
- building_target_recall: `0.4190187841653824`
- building_target_exact: `0.4190187841653824`
- val_q_mean: `0.3516344469040632`
- val_v_mean: `0.364518704606841`
- val_adv_mean: `-0.0744818115606904`

## 指标如何看

- `transition`：一个 1 Hz 的 `(state_t, action_t, reward_t, state_t+1)` 样本，不是一整局比赛。
- `step`：训练的梯度更新次数；报告中的这一行是 held-out `action_mse` 最低的时刻，`best.pt` 就保存自该时刻。
- `action_mse`：12 维归一化战术动作整体的均方误差（旧 10 维 checkpoint 仍兼容），越低越接近留出比赛里的下一步动作。它只适合比较同一角色、同一动作定义和同一数据构建版本，不能横向比较不同兵种强弱。
- `nav_mse`：仅 5 秒子目标位移的归一化均方误差，越低越好；它不是以米为单位的路径误差。
- `fire_acc`：开火/不开火的总准确率。不开火样本很多时会虚高，不能单独使用。
- `fire_f1`：只针对“应该开火”正类的 precision/recall 平衡，越高越好；评估射击陪练优先看它和 `fire_pos_rate`。
- `fire_pos_rate`：留出集中真实开火标签的比例，是数据分布，不是模型得分。很低时 `fire_acc` 几乎没有解释力。
- `target_top1`：预测的最高概率目标是否正确，包含 `<no target>`；无目标很多时会偏高。应优先看 `target_top1_named`。
- `target_top1_named`：只在真实标签为具体敌方单位时的 Top-1 命中率，越高越好；`target_named_rate` 是这种时刻在验证集中的比例。
- `val_loss`、`val_q_mean`、`val_v_mean`、`val_adv_mean`：IQL 的内部优化量，不是胜率。主要用来发现 NaN、数值爆炸或训练/验证持续背离。

判断顺序：先确认数据校验通过，再选择最低 `action_mse` 的 `best.pt`，随后结合 `fire_f1`、`target_top1_named` 和回放场景检查行为。离线指标只能证明对未见日志的复现程度，不能证明比赛强度或在线策略收益。

## Scope

This is an offline behaviour policy learned from referee logs. It is a sparring behaviour module, not a dynamics/world model and not evidence of online competitive strength.
