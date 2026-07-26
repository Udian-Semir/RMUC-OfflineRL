"""Write a compact, inspectable report for one role-specific offline policy."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _best_eval(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    key = "action_mse"
    usable = [row for row in rows if row.get(key) not in (None, "")]
    if not usable:
        return rows[-1]
    return min(usable, key=lambda row: float(row[key]))


def write_report(data_dir: Path, run_dir: Path, role: str) -> Path:
    with (data_dir / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    best = _best_eval(run_dir / "eval_log.csv")
    checkpoint = "best.pt" if (run_dir / "best.pt").exists() else "final.pt"
    check_log = data_dir / "check_log.txt"
    data_check = "not recorded"
    if check_log.exists():
        data_check = "passed" if "OK — dataset looks sane." in check_log.read_text(encoding="utf-8") else "needs review"
    lines = [
        f"# {role} OfflineRL Report",
        "",
        "## Dataset",
        "",
        f"- agent types: `{', '.join(meta.get('agent_types', []))}`",
        f"- action mode: `{meta.get('action_mode')}`",
        f"- observation/action dimensions: `{meta.get('obs_dim')} / {meta.get('action_dim')}`",
        f"- train: `{meta.get('n_train_episodes')}` episodes, `{meta.get('n_train_transitions')}` transitions",
        f"- validation: `{meta.get('n_val_episodes')}` episodes, `{meta.get('n_val_transitions')}` transitions",
        f"- match duration: `{meta.get('match_seconds')} s`",
        f"- dataset sanity check: `{data_check}`",
        "",
        "## Data Semantics",
        "",
        "- Red and blue trajectories are both included; blue trajectories are rotated into the red-centric ego frame before training.",
        "- The validation split is by match, not randomly sampled seconds from the same match.",
        "- Tactical navigation is the normalized 5-second ego displacement. The fire gate comes from the selected weapon's ammunition counter. Target labels are inferred from turret bearing and live enemy geometry because the referee log does not record target IDs.",
        "",
        "## Selected Checkpoint",
        "",
        f"- use `{checkpoint}` for deployment/evaluation",
    ]
    if best:
        lines.extend(["", "## Best Validation Row", ""])
        for key, value in best.items():
            lines.append(f"- {key}: `{value}`")
        if float(best.get("fire_pos_rate", "0") or 0) == 0.0:
            lines.extend([
                "",
                "No positive fire labels are present for this role. This checkpoint is a movement/intent sparring policy, not a shooting policy.",
            ])
    lines.extend([
        "",
        "## 指标如何看",
        "",
        "- `transition`：一个 1 Hz 的 `(state_t, action_t, reward_t, state_t+1)` 样本，不是一整局比赛。",
        "- `step`：训练的梯度更新次数；报告中的这一行是 held-out `action_mse` 最低的时刻，`best.pt` 就保存自该时刻。",
        "- `action_mse`：10 维归一化动作整体的均方误差，越低越接近留出比赛里的下一步动作。它只适合比较同一角色、同一动作定义和同一数据构建版本，不能横向比较不同兵种强弱。",
        "- `nav_mse`：仅 5 秒子目标位移的归一化均方误差，越低越好；它不是以米为单位的路径误差。",
        "- `fire_acc`：开火/不开火的总准确率。不开火样本很多时会虚高，不能单独使用。",
        "- `fire_f1`：只针对“应该开火”正类的 precision/recall 平衡，越高越好；评估射击陪练优先看它和 `fire_pos_rate`。",
        "- `fire_pos_rate`：留出集中真实开火标签的比例，是数据分布，不是模型得分。很低时 `fire_acc` 几乎没有解释力。",
        "- `target_top1`：预测的最高概率目标是否正确，包含 `<no target>`；无目标很多时会偏高。应优先看 `target_top1_named`。",
        "- `target_top1_named`：只在真实标签为具体敌方单位时的 Top-1 命中率，越高越好；`target_named_rate` 是这种时刻在验证集中的比例。",
        "- `val_loss`、`val_q_mean`、`val_v_mean`、`val_adv_mean`：IQL 的内部优化量，不是胜率。主要用来发现 NaN、数值爆炸或训练/验证持续背离。",
        "",
        "判断顺序：先确认数据校验通过，再选择最低 `action_mse` 的 `best.pt`，随后结合 `fire_f1`、`target_top1_named` 和回放场景检查行为。离线指标只能证明对未见日志的复现程度，不能证明比赛强度或在线策略收益。",
        "",
        "## Scope",
        "",
        "This is an offline behaviour policy learned from referee logs. It is a sparring behaviour module, not a dynamics/world model and not evidence of online competitive strength.",
        "",
    ])
    out = run_dir / "ROLE_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--role", required=True)
    args = parser.parse_args()
    out = write_report(Path(args.data), Path(args.run), args.role)
    print(out)


if __name__ == "__main__":
    main()
