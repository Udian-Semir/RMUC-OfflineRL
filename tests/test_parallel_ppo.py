from __future__ import annotations

import unittest

from sentry_tactical_rl.env import SentryTacticalEnv
from sentry_tactical_rl.ppo import PPOConfig, PPOTrainer
from sentry_tactical_rl.semantic_map import SemanticMap


class ParallelPPOTest(unittest.TestCase):
    def test_parallel_rollout_flattens_time_and_environment_axes(self) -> None:
        env = SentryTacticalEnv(
            semantic_map=SemanticMap.demo(),
            horizon=3,
            seed=17,
            sparring_backend="scripted",
            blue_sparring_profiles=("aggressive", "measured"),
        )
        config = PPOConfig(
            num_envs=2,
            worker_torch_threads=1,
            rollout_steps=4,
            epochs=1,
            minibatch_size=8,
        )
        trainer = PPOTrainer(env, config, device="cpu", seed=17)
        try:
            self.assertEqual(trainer.obs["map"].shape[0], 2)
            batch = trainer.collect_rollout()
            self.assertEqual(tuple(batch["maps"].shape[:2]), (8, env.map_channels))
            self.assertEqual(tuple(batch["vectors"].shape), (8, env.vector_dim))
            self.assertEqual(tuple(batch["actions"].shape), (8, 3))
            self.assertEqual(tuple(batch["advantages"].shape), (8,))
            self.assertEqual(tuple(batch["returns"].shape), (8,))
            self.assertEqual(trainer.episodes_finished, 2)
        finally:
            trainer.close()

    def test_step_reports_policy_and_final_published_command(self) -> None:
        env = SentryTacticalEnv(
            semantic_map=SemanticMap.demo(),
            horizon=2,
            seed=23,
            sparring_backend="scripted",
        )
        policy_action = (0, env.NONE_TARGET, env.FIRE_HOLD)
        _, _, _, info = env.step(policy_action)
        self.assertEqual(info["policy_goal_idx"], policy_action[0])
        self.assertEqual(info["policy_target_idx"], policy_action[1])
        self.assertEqual(info["executed_target_idx"], policy_action[1])
        self.assertEqual(info["executed_fire_mode"], policy_action[2])
        self.assertIn("execution_goal_cell", info)
        self.assertIn("execution_goal_xy_m", info)


if __name__ == "__main__":
    unittest.main()
