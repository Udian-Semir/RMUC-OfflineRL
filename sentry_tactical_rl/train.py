"""CLI entry point: ``python -m sentry_tactical_rl.train --config ...``."""
from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from .env import SentryTacticalEnv
from .live_plot import TrainingDashboard
from .ppo import PPOConfig, PPOTrainer
from .semantic_map import SemanticMap


def _device(value: str) -> str:
    return "cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the single-sentry tactical PPO demo")
    parser.add_argument("--config", default="sentry_tactical_rl/configs/demo.yaml")
    parser.add_argument("--map-yaml", default=None, help="replace the synthetic map with a semantic-map YAML")
    parser.add_argument("--map-json", default=None,
                        help="load the aligned RMUC semantic JSON instead of the synthetic map")
    parser.add_argument("--obstacle-map", default=None,
                        help="black/white occupancy image paired with --map-json")
    parser.add_argument("--updates", type=int, default=None, help="override config train.updates")
    parser.add_argument("--out-dir", default=None, help="override train.out_dir for checkpoints and telemetry")
    parser.add_argument("--live", action="store_true",
                        help="open a live reward/cost dashboard; metrics are always saved to CSV")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    env_cfg, train_cfg = config["env"], config["train"]
    seed = int(train_cfg.get("seed", env_cfg.get("seed", 7)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.map_json and args.map_yaml:
        parser.error("use only one of --map-json and --map-yaml")
    if args.map_json:
        semantic_map = SemanticMap.from_aligned_json(args.map_json, obstacle_path=args.obstacle_map)
    else:
        semantic_map = SemanticMap.from_yaml(args.map_yaml) if args.map_yaml else None
    env = SentryTacticalEnv(semantic_map=semantic_map, **env_cfg)
    valid_ppo_fields = {field.name for field in fields(PPOConfig)}
    ppo = PPOConfig(**{key: value for key, value in train_cfg.items() if key in valid_ppo_fields})
    trainer = PPOTrainer(env, ppo, device=_device(str(train_cfg.get("device", "auto"))))
    updates = args.updates or int(train_cfg["updates"])
    out_dir = Path(args.out_dir or train_cfg["out_dir"])
    checkpoint_every = int(train_cfg.get("checkpoint_every", 25))
    print(f"training on {trainer.device}; goals={env.n_goals}, targets={env.n_targets}, vector={env.vector_dim}")
    dashboard = TrainingDashboard(out_dir, live=args.live)
    try:
        for update in range(1, updates + 1):
            metrics = trainer.train_update()
            dashboard.update(update, metrics)
            if update == 1 or update % 5 == 0:
                print(
                    "update={:04d} return={:7.3f} reward={:7.3f} path_cost={:7.3f} "
                    "dmg={:6.2f}/{:6.2f} invalid={:.3f} pi={:7.4f} v={:7.4f} ent={:6.3f} episodes={:.0f}".format(
                        update,
                        metrics["mean_episode_return"],
                        metrics.get("mean_reward", float("nan")),
                        metrics.get("mean_path_cost", float("nan")),
                        metrics.get("mean_damage_dealt", 0.0),
                        metrics.get("mean_damage_taken", 0.0),
                        metrics.get("mean_invalid_action", 0.0),
                        metrics["policy_loss"],
                        metrics["value_loss"],
                        metrics["entropy"],
                        metrics["episodes"],
                    )
                )
            if update % checkpoint_every == 0 or update == updates:
                trainer.save(out_dir / f"ppo_{update:05d}.pt", extra={"update": update, "config": config})
    finally:
        dashboard.close()


if __name__ == "__main__":
    main()
