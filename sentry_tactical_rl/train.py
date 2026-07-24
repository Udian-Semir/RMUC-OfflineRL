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
from .ppo import PPOConfig, PPOTrainer
from .semantic_map import SemanticMap


def _device(value: str) -> str:
    return "cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the single-sentry tactical PPO demo")
    parser.add_argument("--config", default="sentry_tactical_rl/configs/demo.yaml")
    parser.add_argument("--map-yaml", default=None, help="replace the synthetic map with a semantic-map YAML")
    parser.add_argument("--updates", type=int, default=None, help="override config train.updates")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    env_cfg, train_cfg = config["env"], config["train"]
    seed = int(train_cfg.get("seed", env_cfg.get("seed", 7)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    semantic_map = SemanticMap.from_yaml(args.map_yaml) if args.map_yaml else None
    env = SentryTacticalEnv(semantic_map=semantic_map, **env_cfg)
    valid_ppo_fields = {field.name for field in fields(PPOConfig)}
    ppo = PPOConfig(**{key: value for key, value in train_cfg.items() if key in valid_ppo_fields})
    trainer = PPOTrainer(env, ppo, device=_device(str(train_cfg.get("device", "auto"))))
    updates = args.updates or int(train_cfg["updates"])
    out_dir = Path(train_cfg["out_dir"])
    checkpoint_every = int(train_cfg.get("checkpoint_every", 25))
    print(f"training on {trainer.device}; goals={env.n_goals}, targets={env.n_targets}, vector={env.vector_dim}")
    for update in range(1, updates + 1):
        metrics = trainer.train_update()
        if update == 1 or update % 5 == 0:
            print("update={:04d} return={:7.3f} pi={:7.4f} v={:7.4f} ent={:6.3f} episodes={:.0f}".format(
                update, metrics["mean_episode_return"], metrics["policy_loss"], metrics["value_loss"],
                metrics["entropy"], metrics["episodes"],
            ))
        if update % checkpoint_every == 0 or update == updates:
            trainer.save(out_dir / f"ppo_{update:05d}.pt", extra={"update": update, "config": config})


if __name__ == "__main__":
    main()
