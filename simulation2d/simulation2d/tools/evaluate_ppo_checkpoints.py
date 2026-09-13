"""Evaluate saved PPO checkpoints without rendering videos.

The trainer's rollout return is intentionally not used as a checkpoint
selection result: it mixes exploration actions with whichever blue sparring
profile happened to be sampled.  This tool freezes each checkpoint and runs a
fixed seed/profile suite before one representative full-field video is made.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..world.environment import SentryTacticalEnv
from ..policy.network import TacticalActorCritic
from ..world.semantic_map import SemanticMap


def _device(value: str) -> str:
    return "cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value)


def _csv_values(value: str, convert):
    return tuple(convert(piece.strip()) for piece in value.split(",") if piece.strip())


def _obs_tensors(obs: dict[str, np.ndarray], device: torch.device):
    return tuple(torch.as_tensor(obs[key], device=device).unsqueeze(0) for key in ("map", "vector", "goal_mask", "target_mask"))


def _one_episode(model: TacticalActorCritic, env: SentryTacticalEnv, *, seed: int,
                 device: torch.device) -> dict[str, float | str]:
    obs = env.reset(seed=seed)
    total = sentry_robot = sentry_outpost = sentry_base = taken = 0.0
    red_outpost_damage = red_base_damage = invalid = 0.0
    done = False
    while not done:
        with torch.no_grad():
            action, _, _ = model.act(*_obs_tensors(obs, device), deterministic=True)
        obs, reward, done, info = env.step(action.squeeze(0).cpu().numpy())
        total += float(reward)
        sentry_robot += float(info.get("sentry_robot_damage", info.get("damage_dealt", 0.0)))
        sentry_outpost += float(info.get("sentry_blue_outpost_damage", 0.0))
        sentry_base += float(info.get("sentry_blue_base_damage", 0.0))
        taken += float(info.get("damage_taken", 0.0))
        red_outpost_damage += float(info.get("red_outpost_damage", 0.0))
        red_base_damage += float(info.get("red_base_damage", 0.0))
        invalid += float(bool(info.get("invalid_action", False)))
    return {
        "return": total,
        "sentry_robot_damage": sentry_robot,
        "sentry_outpost_damage": sentry_outpost,
        "sentry_base_damage": sentry_base,
        "damage_taken": taken,
        "red_outpost_damage": red_outpost_damage,
        "red_base_damage": red_base_damage,
        "invalid_actions": invalid,
        "red_base_hp": float(env.match.red.base_hp),
        "red_outpost_hp": float(env.match.red.outpost_hp),
        "outcome": str(info.get("outcome", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-suite PPO checkpoint evaluator")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", default=None,
                        help="optional comma-separated checkpoint filenames; defaults to every ppo_*.pt")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--map-json", type=Path, required=True)
    parser.add_argument("--obstacle-map", type=Path, required=True)
    parser.add_argument("--seeds", default="1001,1002", help="comma-separated held-out episode seeds")
    parser.add_argument("--profiles", default="aggressive,pressure,measured")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    checkpoints = sorted(args.run_dir.glob("ppo_*.pt"))
    if args.checkpoints:
        names = _csv_values(args.checkpoints, str)
        checkpoints = [args.run_dir / name for name in names]
        missing = [path for path in checkpoints if not path.is_file()]
        if missing:
            raise SystemExit(f"checkpoint files not found: {missing}")
    if not checkpoints:
        raise SystemExit(f"no ppo_*.pt files in {args.run_dir}")
    seeds = _csv_values(args.seeds, int)
    profiles = _csv_values(args.profiles, str)
    if not seeds or not profiles:
        raise SystemExit("--seeds and --profiles must not be empty")

    map_obj = SemanticMap.from_aligned_json(args.map_json, obstacle_path=args.obstacle_map)
    device = torch.device(_device(args.device if args.device != "auto" else str(config["train"].get("device", "auto"))))
    shape_env = SentryTacticalEnv(semantic_map=map_obj, **dict(config["env"]))
    case_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        payload = torch.load(checkpoint_path, map_location="cpu")
        env_cfg = dict(config["env"])
        env_cfg["sparring_backend"] = "offline"
        model = TacticalActorCritic(
            shape_env.map_channels, shape_env.vector_dim, shape_env.n_goals, shape_env.n_targets,
        ).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        rows = []
        for profile in profiles:
            env_cfg["blue_sparring_profiles"] = [profile]
            env = SentryTacticalEnv(semantic_map=map_obj, **env_cfg)
            for seed in seeds:
                row = _one_episode(model, env, seed=seed, device=device)
                row.update(checkpoint=checkpoint_path.name, seed=seed, profile=profile)
                rows.append(row)
                case_rows.append(row)
        numeric = ("return", "sentry_robot_damage", "sentry_outpost_damage", "sentry_base_damage",
                   "damage_taken", "red_outpost_damage", "red_base_damage", "invalid_actions",
                   "red_base_hp", "red_outpost_hp")
        summary = {key: float(np.mean([float(row[key]) for row in rows])) for key in numeric}
        summary.update(
            checkpoint=checkpoint_path.name,
            cases=len(rows),
            win_rate=float(np.mean([row["outcome"] == "red_win" for row in rows])),
            sentry_objective_damage=summary["sentry_robot_damage"] + summary["sentry_outpost_damage"] + summary["sentry_base_damage"],
        )
        summary_rows.append(summary)
        print(f"finished checkpoint={checkpoint_path.name} cases={len(rows)} "
              f"win_rate={summary['win_rate']:.3f} return={summary['return']:.2f}", flush=True)

    # This is a declared evaluation rank, not the training reward.  It first
    # refuses to trade away a match/base result for easy damage, then uses the
    # red sentry's own contribution (not teammate damage) and return as ties.
    summary_rows.sort(key=lambda row: (
        -row["win_rate"], -row["red_base_hp"], -row["red_outpost_hp"],
        -row["sentry_objective_damage"], -row["return"], row["invalid_actions"],
    ))
    for rank, row in enumerate(summary_rows, start=1):
        row["rank"] = rank
    out_dir = args.out or args.run_dir / "fixed_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("checkpoint_cases.csv", case_rows), ("checkpoint_summary.csv", summary_rows)):
        with (out_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    best = summary_rows[0]
    (out_dir / "BEST_CHECKPOINT.txt").write_text(
        f"{best['checkpoint']}\nrank={best['rank']} cases={best['cases']} win_rate={best['win_rate']:.3f}\n",
        encoding="utf-8",
    )
    print(f"evaluation_dir={out_dir}")
    print(f"best={best['checkpoint']} rank=1 win_rate={best['win_rate']:.3f} "
          f"base_hp={best['red_base_hp']:.1f} outpost_hp={best['red_outpost_hp']:.1f} "
          f"sentry_objective={best['sentry_objective_damage']:.1f} return={best['return']:.2f}")


if __name__ == "__main__":
    main()
