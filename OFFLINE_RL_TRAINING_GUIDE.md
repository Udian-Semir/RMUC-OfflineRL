# OfflineRL 训练结果阅读说明

## `transition` 是什么

本仓库的裁判日志采样频率为 1 Hz。一个 transition 是从第 `t` 秒到第 `t+1` 秒的一条监督/强化学习样本：

```text
state_t -> action_t -> reward_t, state_(t+1)
```

`state_t` 包含己方与敌方单位的血量、位置、建筑血量、时间、经济和能力状态；`action_t` 是该角色的 5 秒子目标位移、开火许可和目标优先级；`reward_t` 是离线构造的战术奖励。它不是一局比赛。一局 420 秒的连续轨迹通常提供约 419 条 transition；一个角色数据集会汇集很多场比赛、红蓝双方的此类轨迹。

## 先看数据是否可信

每个 `data/<role>_tactical/check_log.txt` 都应以 `OK — dataset looks sane.` 结束。它检查：维度是否正确、是否有 NaN/Inf、存活时刻是否有移动、开火/目标标签是否异常塌缩，以及不该恒定的观测是否没有接入。

红蓝双方的同角色轨迹都会参与训练。蓝方轨迹先旋转 180 度到同一自我视角，因此一个角色模型可以放到任一阵营；`blue_sentry_iql_tactical` 只是表示该 checkpoint 用于蓝方陪练槽位，不表示只学习蓝方比赛记录。训练/验证按整场比赛划分，不会把同一局的不同时刻同时放入两边。

## 每个数字代表什么

| 字段 | 含义 | 正确看法 |
| --- | --- | --- |
| `transition` | 一个 `(state_t, action_t, reward_t, state_t+1)` 样本 | 数量反映数据规模，不等同于比赛局数。 |
| `step` | 一次梯度更新计数 | 报告选择 validation `action_mse` 最小的 step。 |
| `action_mse` | 10 维归一化动作误差 | 越低越接近未见比赛动作；只比较同一角色/动作定义。 |
| `nav_mse` | 5 秒子目标位移两维的归一化误差 | 越低越好，不是米或路径长度。 |
| `fire_acc` | 开火/不开火总正确率 | 类别不平衡时会虚高，不能单看。 |
| `fire_f1` | 开火正类的 precision/recall 平衡 | 射击行为优先看，越高越好。 |
| `fire_pos_rate` | 真实开火标签比例 | 数据分布，不是模型成绩；越低时 `fire_acc` 越不可信。 |
| `target_top1` | 最高概率目标的正确率，含无目标 | 无目标比例大时会虚高。 |
| `target_top1_named` | 有具体敌方目标时的 Top-1 正确率 | 评估选敌能力应优先看，越高越好。 |
| `target_named_rate` | 真实标签为具体敌人的比例 | 用来判断 `target_top1_named` 覆盖的样本量。 |
| `val_loss` / `val_q_mean` / `val_v_mean` / `val_adv_mean` | IQL 内部目标和价值统计 | 不代表胜率；只看是否 NaN、爆炸或和验证误差明显背离。 |

## 如何选择与验收

1. 只用每个运行目录里的 `best.pt`，不要默认使用 `final.pt`。前者按验证集 `action_mse` 选择，后者是停止时的最后权重，可能已经过拟合。
2. 移动是否像对应角色：看 `action_mse`、`nav_mse` 和连续回放轨迹。
3. 是否会在合适时机射击：看 `fire_f1`，并先看 `fire_pos_rate` 是否足够大；不能用 `fire_acc` 下结论。
4. 是否会选对敌人：看 `target_top1_named`，同时看 `target_named_rate`。目标标签由枪口朝向和敌方几何关系推断，不是裁判提供的真 target ID。
5. 最终必须在 2D 规则仿真中回放不同前哨、基地血量、队友位置和对手风格的场景。离线指标只证明模仿未见日志，不能证明陪练已经能赢比赛，更不能替代红方主哨兵的在线 PPO 评估。

工程角色在当前官方日志中没有正开火标签，所以它的 checkpoint 只用于移动/意图陪练；其 `fire_acc=1.0` 不具有战斗能力含义。
