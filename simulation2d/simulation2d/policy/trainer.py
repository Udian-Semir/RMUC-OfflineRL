"""Small, dependency-free PPO implementation for the tactical demo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..world.environment import SentryTacticalEnv
from .network import TacticalActorCritic
from .parallel_env import ParallelSentryEnvPool


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
    num_envs: int = 1
    worker_torch_threads: int = 1


class PPOTrainer:
    def __init__(
        self,
        env: SentryTacticalEnv,
        config: PPOConfig,
        device: str | torch.device = "cpu",
        *,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.config = config
        self.device = torch.device(device)
        if config.num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        self.env_pool = (
            ParallelSentryEnvPool(
                env,
                config.num_envs,
                base_seed=seed,
                worker_torch_threads=config.worker_torch_threads,
            )
            if config.num_envs > 1 else None
        )
        self.num_envs = config.num_envs
        self.model = TacticalActorCritic(env.map_channels, env.vector_dim, env.n_goals, env.n_targets).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.obs = self.env_pool.observations() if self.env_pool is not None else self._batch_obs(env.reset())
        self.episodes_finished = 0
        self.episode_returns: list[float] = []
        self._running_return = np.zeros(self.num_envs, dtype=np.float64)
        self.last_rollout_stats: dict[str, float] = {}

    @staticmethod
    def _batch_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: np.expand_dims(value, axis=0) for key, value in obs.items()}

    def _tensor_obs(self, obs: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        maps = torch.as_tensor(obs["map"], device=self.device)
        vectors = torch.as_tensor(obs["vector"], device=self.device)
        goals = torch.as_tensor(obs["goal_mask"], device=self.device)
        targets = torch.as_tensor(obs["target_mask"], device=self.device)
        return maps, vectors, goals, targets

    def _step_envs(
        self, actions: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if self.env_pool is not None:
            return self.env_pool.step(actions)
        next_obs, reward, done, info = self.env.step(actions[0])
        if done:
            next_obs = self.env.reset()
        return (
            self._batch_obs(next_obs),
            np.asarray([reward], dtype=np.float32),
            np.asarray([done], dtype=np.bool_),
            [info],
        )

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
            "sentry_blue_outpost_damage": [],
            "sentry_blue_base_damage": [],
            "ally_blue_outpost_damage": [],
            "ally_blue_base_damage": [],
            "blue_outpost_damage": [],
            "red_outpost_damage": [],
            "blue_base_damage": [],
            "red_base_damage": [],
            "healing": [],
            "invalid_action": [],
            "goal_switch": [],
            "goal_switch_blocked": [],
            "blue_autoaim_probability": [],
        }
        for _ in range(self.config.rollout_steps):
            maps, vectors, goals, targets = self._tensor_obs(self.obs)
            with torch.no_grad():
                action, log_prob, value = self.model.act(maps, vectors, goals, targets)
            action_np = action.cpu().numpy()
            current_obs = {key: source.copy() for key, source in self.obs.items()}
            next_obs, rewards, dones, infos = self._step_envs(action_np)
            telemetry_keys = {
                "reward": lambda info, index: rewards[index],
                "path_cost": lambda info, index: info.get("path_cost", 0.0),
                "path_risk": lambda info, index: info.get("path_risk", 0.0),
                "damage_dealt": lambda info, index: info.get("damage_dealt", 0.0),
                "damage_taken": lambda info, index: info.get("damage_taken", 0.0),
                "sentry_blue_outpost_damage": lambda info, index: info.get("sentry_blue_outpost_damage", 0.0),
                "sentry_blue_base_damage": lambda info, index: info.get("sentry_blue_base_damage", 0.0),
                "ally_blue_outpost_damage": lambda info, index: info.get("ally_blue_outpost_damage", 0.0),
                "ally_blue_base_damage": lambda info, index: info.get("ally_blue_base_damage", 0.0),
                "blue_outpost_damage": lambda info, index: info.get("blue_outpost_damage", 0.0),
                "red_outpost_damage": lambda info, index: info.get("red_outpost_damage", 0.0),
                "blue_base_damage": lambda info, index: info.get("blue_base_damage", 0.0),
                "red_base_damage": lambda info, index: info.get("red_base_damage", 0.0),
                "healing": lambda info, index: info.get("reward_terms", {}).get("healing", 0.0),
                "invalid_action": lambda info, index: bool(info.get("invalid_action", False)),
                "goal_switch": lambda info, index: bool(info.get("goal_switch", False)),
                "goal_switch_blocked": lambda info, index: bool(info.get("goal_switch_blocked", False)),
                "blue_autoaim_probability": lambda info, index: info.get("blue_autoaim_probability", 0.0),
            }
            for index, info in enumerate(infos):
                for key, getter in telemetry_keys.items():
                    telemetry[key].append(float(getter(info, index)))
            for key, source in (
                ("map", current_obs["map"]),
                ("vector", current_obs["vector"]),
                ("goal_mask", current_obs["goal_mask"]),
                ("target_mask", current_obs["target_mask"]),
                ("action", action_np),
                ("log_prob", log_prob.cpu().numpy()),
                ("value", value.cpu().numpy()),
                ("reward", rewards),
                ("done", dones),
            ):
                records[key].append(source)
            self._running_return += rewards
            for index in np.flatnonzero(dones):
                self.episode_returns.append(float(self._running_return[index]))
                self.episodes_finished += 1
                self._running_return[index] = 0.0
            self.obs = next_obs

        self.last_rollout_stats = {
            f"mean_{key}": float(np.mean(values)) if values else 0.0
            for key, values in telemetry.items()
        }

        with torch.no_grad():
            bootstrap = self.model.act(*self._tensor_obs(self.obs), deterministic=True)[2].cpu().numpy()
        rewards = np.asarray(records["reward"], dtype=np.float32)
        values = np.asarray(records["value"], dtype=np.float32)
        dones = np.asarray(records["done"], dtype=np.float32)
        advantages = np.zeros_like(rewards)
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        next_value = bootstrap
        for t in range(len(rewards) - 1, -1, -1):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * next_value * nonterminal - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
            next_value = values[t]
        returns = advantages + values
        def flatten_time_env(values: Any) -> np.ndarray:
            array = np.asarray(values)
            return array.reshape((-1, *array.shape[2:]))

        batch = {
            "maps": torch.as_tensor(flatten_time_env(records["map"]), device=self.device),
            "vectors": torch.as_tensor(flatten_time_env(records["vector"]), device=self.device),
            "goal_masks": torch.as_tensor(flatten_time_env(records["goal_mask"]), device=self.device),
            "target_masks": torch.as_tensor(flatten_time_env(records["target_mask"]), device=self.device),
            "actions": torch.as_tensor(flatten_time_env(records["action"]), device=self.device, dtype=torch.long),
            "old_log_probs": torch.as_tensor(flatten_time_env(records["log_prob"]), device=self.device),
            "advantages": torch.as_tensor(advantages.reshape(-1), device=self.device),
            "returns": torch.as_tensor(returns.reshape(-1), device=self.device),
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

    def load(self, path: str | Path, *, load_optimizer: bool = True) -> dict[str, Any]:
        """Restore a PPO checkpoint and return its metadata.

        Loading the optimizer is the default for genuine continuation training;
        callers can disable it for policy-only evaluation or fine-tuning.
        """
        payload = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(payload["model"])
        if load_optimizer and payload.get("optimizer"):
            self.optimizer.load_state_dict(payload["optimizer"])
        return dict(payload.get("extra") or {})

    def close(self) -> None:
        if self.env_pool is not None:
            self.env_pool.close()
