"""Small, dependency-free PPO implementation for the tactical demo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .env import SentryTacticalEnv
from .model import TacticalActorCritic


@dataclass
class PPOConfig:
    rollout_steps: int = 256
    epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.8


class PPOTrainer:
    def __init__(self, env: SentryTacticalEnv, config: PPOConfig, device: str | torch.device = "cpu") -> None:
        self.env = env
        self.config = config
        self.device = torch.device(device)
        self.model = TacticalActorCritic(env.map_channels, env.vector_dim, env.n_goals, env.n_targets).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.obs = env.reset()
        self.episodes_finished = 0
        self.episode_returns: list[float] = []
        self._running_return = 0.0
        self.last_rollout_stats: dict[str, float] = {}

    def _tensor_obs(self, obs: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        maps = torch.as_tensor(obs["map"], device=self.device).unsqueeze(0)
        vectors = torch.as_tensor(obs["vector"], device=self.device).unsqueeze(0)
        goals = torch.as_tensor(obs["goal_mask"], device=self.device).unsqueeze(0)
        targets = torch.as_tensor(obs["target_mask"], device=self.device).unsqueeze(0)
        return maps, vectors, goals, targets

    def collect_rollout(self) -> dict[str, torch.Tensor]:
        records: dict[str, list[Any]] = {key: [] for key in (
            "map", "vector", "goal_mask", "target_mask", "action", "log_prob", "value", "reward", "done"
        )}
        telemetry: dict[str, list[float]] = {
            "reward": [],
            "path_cost": [],
            "path_risk": [],
            "damage_dealt": [],
            "damage_taken": [],
            "blue_outpost_damage": [],
            "red_outpost_damage": [],
            "blue_base_damage": [],
            "red_base_damage": [],
            "healing": [],
            "invalid_action": [],
            "goal_switch": [],
            "goal_switch_blocked": [],
        }
        for _ in range(self.config.rollout_steps):
            maps, vectors, goals, targets = self._tensor_obs(self.obs)
            with torch.no_grad():
                action, log_prob, value = self.model.act(maps, vectors, goals, targets)
            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, done, info = self.env.step(action_np)
            telemetry["reward"].append(float(reward))
            telemetry["path_cost"].append(float(info.get("path_cost", 0.0)))
            telemetry["path_risk"].append(float(info.get("path_risk", 0.0)))
            telemetry["damage_dealt"].append(float(info.get("damage_dealt", 0.0)))
            telemetry["damage_taken"].append(float(info.get("damage_taken", 0.0)))
            telemetry["blue_outpost_damage"].append(float(info.get("blue_outpost_damage", 0.0)))
            telemetry["red_outpost_damage"].append(float(info.get("red_outpost_damage", 0.0)))
            telemetry["blue_base_damage"].append(float(info.get("blue_base_damage", 0.0)))
            telemetry["red_base_damage"].append(float(info.get("red_base_damage", 0.0)))
            telemetry["healing"].append(float(info.get("reward_terms", {}).get("healing", 0.0)))
            telemetry["invalid_action"].append(float(bool(info.get("invalid_action", False))))
            telemetry["goal_switch"].append(float(bool(info.get("goal_switch", False))))
            telemetry["goal_switch_blocked"].append(float(bool(info.get("goal_switch_blocked", False))))
            for key, source in (("map", self.obs["map"]), ("vector", self.obs["vector"]),
                                ("goal_mask", self.obs["goal_mask"]), ("target_mask", self.obs["target_mask"]),
                                ("action", action_np), ("log_prob", log_prob.item()), ("value", value.item()),
                                ("reward", reward), ("done", done)):
                records[key].append(source)
            self._running_return += reward
            if done:
                self.episode_returns.append(self._running_return)
                self.episodes_finished += 1
                self._running_return = 0.0
                self.obs = self.env.reset()
            else:
                self.obs = next_obs

        self.last_rollout_stats = {
            f"mean_{key}": float(np.mean(values)) if values else 0.0
            for key, values in telemetry.items()
        }

        with torch.no_grad():
            bootstrap = self.model.act(*self._tensor_obs(self.obs), deterministic=True)[2].item()
        rewards = np.asarray(records["reward"], dtype=np.float32)
        values = np.asarray(records["value"], dtype=np.float32)
        dones = np.asarray(records["done"], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        next_value = bootstrap
        for t in range(len(rewards) - 1, -1, -1):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * next_value * nonterminal - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
            next_value = values[t]
        returns = advantages + values
        batch = {
            "maps": torch.as_tensor(np.asarray(records["map"]), device=self.device),
            "vectors": torch.as_tensor(np.asarray(records["vector"]), device=self.device),
            "goal_masks": torch.as_tensor(np.asarray(records["goal_mask"]), device=self.device),
            "target_masks": torch.as_tensor(np.asarray(records["target_mask"]), device=self.device),
            "actions": torch.as_tensor(np.asarray(records["action"]), device=self.device, dtype=torch.long),
            "old_log_probs": torch.as_tensor(np.asarray(records["log_prob"]), device=self.device),
            "advantages": torch.as_tensor(advantages, device=self.device),
            "returns": torch.as_tensor(returns, device=self.device),
        }
        batch["advantages"] = (batch["advantages"] - batch["advantages"].mean()) / (batch["advantages"].std() + 1e-8)
        return batch

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        n = batch["actions"].shape[0]
        losses: list[tuple[float, float, float]] = []
        for _ in range(self.config.epochs):
            for indices in torch.randperm(n, device=self.device).split(self.config.minibatch_size):
                log_prob, entropy, value = self.model.evaluate_actions(
                    batch["maps"][indices], batch["vectors"][indices], batch["goal_masks"][indices],
                    batch["target_masks"][indices], batch["actions"][indices],
                )
                ratio = (log_prob - batch["old_log_probs"][indices]).exp()
                advantage = batch["advantages"][indices]
                policy_loss = -torch.minimum(ratio * advantage,
                                             torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantage).mean()
                value_loss = (value - batch["returns"][indices]).square().mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                losses.append((float(policy_loss.detach()), float(value_loss.detach()), float(entropy_loss.detach())))
        means = np.mean(losses, axis=0)
        return {"policy_loss": float(means[0]), "value_loss": float(means[1]), "entropy": float(means[2])}

    def train_update(self) -> dict[str, float]:
        batch = self.collect_rollout()
        metrics = self.update(batch)
        recent = self.episode_returns[-10:]
        metrics["mean_episode_return"] = float(np.mean(recent)) if recent else float("nan")
        metrics["episodes"] = float(self.episodes_finished)
        metrics.update(self.last_rollout_stats)
        return metrics

    def save(self, path: str | Path, *, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "extra": extra or {}}, path)
