# 多风格 OfflineRL 陪练池

这里的 checkpoint 只控制红方 PPO 哨兵以外的陪练角色。红方主哨兵永远由在线 PPO 控制，不放入本目录。

目录结构：

```text
datasets/<style>/<role>/       按整局/阵营切分的训练与验证数据
checkpoints/<style>/<role>/    best.pt、日志与角色报告
reports/<style>/<role>/        后续 held-out 回放与风格验收
replays/<style>/<role>/        可视化回放
style_assignments.csv          每个 game/camp 的可审计风格标签
dataset_manifest.csv           每个角色/风格的数据量
```

初始风格由官方日志中的整局统计定义：`pressure` 重视敌建筑伤害和前压，`aggressive` 重视交火密度和前压，`measured` 为其余较稳健局面。它们是数据驱动的互斥初始划分，不等价于穷尽真实赛场风格。

构建：

```bash
python -m rm_rl.data.build_sparring_style_pool \
  --db dataset/rmuc_2026_region_dataset.sqlite --out sparring_style_pool
```

训练全部角色：

```bash
bash scripts/train_sparring_style_pool.sh sparring_style_pool
```

验收时应训练于两个风格、评测于被留出的第三风格；并检查路线、选敌、开火和建筑交互分布是否真的不同。不能仅改命中率/开火概率就称为多风格陪练。
