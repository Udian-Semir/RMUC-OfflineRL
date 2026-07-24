"""Fast environment contract check, no training required."""
from __future__ import annotations

import numpy as np

from .env import SentryTacticalEnv


def main() -> None:
    env = SentryTacticalEnv(horizon=12, seed=11)
    obs = env.reset()
    total = 0.0
    for _ in range(12):
        goals = np.flatnonzero(obs["goal_mask"])
        targets = np.flatnonzero(obs["target_mask"])
        action = (int(goals[0]), int(targets[0]), SentryTacticalEnv.FIRE_ENGAGE)
        obs, reward, done, info = env.step(action)
        total += reward
        if done:
            break
    assert obs["map"].shape == (env.map_channels, env.map.height, env.map.width)
    assert obs["vector"].shape == (env.vector_dim,)
    print(f"smoke ok: return={total:.3f}, steps={env.step_count}, outcome={info.get('outcome', 'running')}")
    print(env.render_ascii())


if __name__ == "__main__":
    main()
