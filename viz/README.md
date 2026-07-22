# viz — 把这个项目"看"出来

四个工具,全部直接读官方数据集和训练好的策略,不依赖任何中间产物。

| 产物 | 脚本 | 说明 |
|---|---|---|
| 交互式决策回放 | `export_replay.py` + `replay.html` + `serve.py` | 浏览器里播放一整场比赛,叠加模型每秒的决策 |
| 哨兵 vs 步兵占位图 | `heatmaps.py` | 项目核心论点的证据图 |
| 策略向量场 | `policy_field.py` | 冻结一帧,问策略"站在场上任何位置你会怎么做" |

前置:先构建两个先验(只需一次),否则策略的 161 维输入会有 12 列是零,决策与训练时不一致。

```bash
python -m rm_rl.data.vis_map    --db dataset/rmuc_2026_region_dataset.sqlite --out data/vis_map.npz
python -m rm_rl.data.team_prior --db dataset/rmuc_2026_region_dataset.sqlite --out data/team_prior.json
```

---

## 1. 交互式回放

```bash
# 看看哪几场打得最凶
python -m viz.export_replay --db dataset/rmuc_2026_region_dataset.sqlite --list

# 导出一场(默认为所有人类驾驶兵种都算一条决策轨迹)
python -m viz.export_replay --db dataset/rmuc_2026_region_dataset.sqlite \
    --game-id 1780384424933 --run rm_runs/infantry_iql_tactical \
    --out viz/replays/game_1780384424933.json

# 起服务(浏览器不允许 file:// 下 fetch,直接双击 html 会是空场地)
python -m viz.serve
```

页面里能做的事:

* 空格播放/暂停,方向键逐秒步进(Shift 加速 10 秒)
* 切换 ego —— 红/蓝双方的步兵3、步兵4、哨兵、英雄各一条轨迹
* **青色实箭头** = 模型的导航子目标,**黄色虚箭头** = 人类实际走向
* 虚线连到模型选中的目标;命中开火许可时车体外圈脉冲
* 底部"决策带":上排人类、下排模型,颜色 = 目标类别,白色小刻度 = 开火许可,
  最下面一行红色标记出两者不一致的秒 —— 那些才是值得停下来看的帧

导出时会打印每条轨迹的一致率。上面那场:

```
红步兵3  target 84.7%  fire 85.6%  nav_cos +0.403
蓝步兵3  target 89.7%  fire 91.9%  nav_cos +0.617
红哨兵   target 76.6%  fire 81.6%  nav_cos +0.246
红英雄   target  2.2%  fire  0.0%  nav_cos +0.002
```

英雄那一行不是 bug,是**故意保留的反例**:英雄是 42 mm、热量上限 50/240,与步兵不同构,
迁移必然失败。留着它,是为了说明"观测层与角色无关"并不等于"什么角色都能迁移"——
只有结构等价才成立,而哨兵恰好等价。

## 2. 占位图

```bash
python -m viz.heatmaps --db dataset/rmuc_2026_region_dataset.sqlite --out viz/figures
```

统计口径是**单场**、1 m 网格(全场 420 格),不是把 613 场叠在一起。这点很重要:
把所有比赛池化会回答"全联盟的哨兵都站在哪",那当然是散开的,恰好把要论证的东西盖掉了。
要说的是"**一场之内**,哨兵几乎不动"。

全数据集(613 场)结果:

| | 单场覆盖 | 位置熵 | 前五格占比 | 跨队相似度 |
|---|---|---|---|---|
| 哨兵 | 25.4 / 420 格 | 1.76 nats | 81.4% | **0.20** |
| 步兵 | 73.7 / 420 格 | 3.50 nats | 40.4% | **0.51** |

输出四张图:

* `per_team.png` —— 同一支队伍,上排哨兵下排步兵。整个论点一张图说完
* `dispersion.png` —— 单场覆盖格数的分布,证明这不是均值把戏
* `similarity.png` —— 跨队走位相似度
* `engagement.png` —— 经验交战图(视线代理)

## 3. 策略向量场

```bash
# 单帧
python -m viz.policy_field --db dataset/....sqlite --game-id 1780384424933 --t 90

# 动图
python -m viz.policy_field --db dataset/....sqlite --game-id 1780384424933 \
    --t0 60 --t1 400 --step 3 --fps 14 --fmt mp4

# 不画图,只量化"这个场到底有多依赖状态"
python -m viz.policy_field --db dataset/....sqlite --game-id 1780384424933 --report
```

做法:冻结某一秒的真实战况,把 ego 依次放到全场 840 个候选位置上各查一次策略。
**不能**只改观测里的 ego 坐标两列 —— 友军相对位移、每个敌人的方位与距离、
交战先验全都依赖 ego 位置,所以合成位置必须走一遍 `features.build_obs`。

这张图除了好看,还回答了一个真问题。`eval/win_alignment` 里,一个**与状态无关的常量策略**
就吃掉了 +0.0265 一致率提升中的 +0.0260,所以"策略与胜者更像"基本是混淆。
`--report` 直接量化学到的策略离常量有多远:

```
spatial  arrow-direction circular SD across the arena :   70.3 deg
temporal target flips per 3s, share of cells          :  34.7%
temporal arrow swing per 3s                           :   32.2 deg
```

常量策略这三项都会是 0。

---

## 关于数据

`viz/replays/*.json` 是一整场比赛逐秒状态的机器可读再编码,等同于分发数据集切片,
因此已在 `.gitignore` 中排除,请各自本地生成。`viz/figures/*.png` 是聚合统计图,随仓库提交;
视频体积大且可复现,同样不提交。
