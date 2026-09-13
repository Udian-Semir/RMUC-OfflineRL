"""ROS-independent real-time driver for the interactive tactical world."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..world.environment import SentryTacticalEnv


Observation = dict[str, np.ndarray]
Action = tuple[int, int, int]


class ActionPolicy(Protocol):
    name: str

    def act(self, observation: Observation) -> Action:
        """Return goal, target, and fire-mode indices."""


class PursuitPolicy:
    """Deterministic fallback that exercises target, pursuit, and damage feedback."""

    name = "pursuit"

    def __init__(self, env: SentryTacticalEnv) -> None:
        self._env = env

    def act(self, observation: Observation) -> Action:
        target_mask = observation["target_mask"]
        candidates = [
            index for index in range(self._env.ROBOT_TARGETS)
            if bool(target_mask[index]) and self._env.enemies[index].role != "aerial"
        ]
        if candidates:
            target_index = min(
                candidates,
                key=lambda index: self._env._distance(
                    self._env.sentry.cell, self._env.enemies[index].cell),
            )
            target_cell = self._env.enemies[target_index].cell
            goal_index = min(
                np.flatnonzero(observation["goal_mask"]),
                key=lambda index: self._env._distance(
                    self._env.map.anchors[self._env.anchor_names[int(index)]], target_cell),
            )
            return int(goal_index), int(target_index), self._env.FIRE_ENGAGE

        valid_goals = np.flatnonzero(observation["goal_mask"])
        return int(valid_goals[0]), self._env.NONE_TARGET, self._env.FIRE_HOLD


class PpoPolicy:
    """Load one TacticalActorCritic checkpoint without coupling ROS to PyTorch."""

    name = "ppo"

    def __init__(
        self,
        env: SentryTacticalEnv,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        deterministic: bool = True,
    ) -> None:
        import torch

        from ..policy.network import TacticalActorCritic

        self._torch = torch
        self._device = torch.device(device)
        self._deterministic = bool(deterministic)
        self._model = TacticalActorCritic(
            env.map_channels, env.vector_dim, env.n_goals, env.n_targets,
        ).to(self._device)
        payload = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location=self._device,
            weights_only=True,
        )
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError("PPO checkpoint does not contain a model state_dict")
        self._model.load_state_dict(payload["model"])
        self._model.eval()

    def act(self, observation: Observation) -> Action:
        torch = self._torch
        with torch.inference_mode():
            action, _, _ = self._model.act(
                torch.as_tensor(observation["map"][None], device=self._device),
                torch.as_tensor(observation["vector"][None], device=self._device),
                torch.as_tensor(observation["goal_mask"][None], device=self._device),
                torch.as_tensor(observation["target_mask"][None], device=self._device),
                deterministic=self._deterministic,
            )
        return tuple(int(value) for value in action[0].cpu().tolist())


@dataclass(frozen=True)
class LiveStep:
    episode: int
    action: Action
    observation: Observation
    reward: float
    done: bool
    info: dict[str, object]


class LiveSimulationEngine:
    """Own environment reset/step state for a wall-timer or simulation clock."""

    def __init__(
        self,
        env: SentryTacticalEnv,
        policy: ActionPolicy,
        *,
        seed: int,
        loop_episodes: bool = True,
    ) -> None:
        self.env = env
        self.policy = policy
        self.seed = int(seed)
        self.loop_episodes = bool(loop_episodes)
        self.episode = 0
        self.observation = env.reset(seed=self.seed)
        self.finished = False

    def reset(self) -> Observation:
        self.episode += 1
        self.observation = self.env.reset(seed=self.seed + self.episode)
        self.finished = False
        return self.observation

    def set_external_sentry_state(self, **state: float | bool) -> None:
        """Use the real Gazebo sentry pose for the next policy decision."""
        setter = getattr(self.env, "set_external_sentry_state", None)
        if setter is None:
            return
        setter(**state)
        self.observation = self.env.observe()

    def clear_external_sentry_state(self) -> None:
        clearer = getattr(self.env, "clear_external_sentry_state", None)
        if clearer is not None:
            clearer()

    def step(self) -> LiveStep:
        if self.finished:
            raise RuntimeError("simulation episode is complete; reset before stepping again")
        action = self.policy.act(self.observation)
        observation, reward, done, info = self.env.step(action)
        result = LiveStep(
            episode=self.episode,
            action=action,
            observation=observation,
            reward=float(reward),
            done=bool(done),
            info=info,
        )
        self.observation = observation
        if done:
            if self.loop_episodes:
                self.reset()
            else:
                self.finished = True
        return result
