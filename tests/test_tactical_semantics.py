"""Tests that semantic geometry is usable by the tactical rule environment."""
from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from sentry_tactical_rl.env import SentryTacticalEnv
from sentry_tactical_rl.semantic_map import SemanticMap
from sentry_tactical_rl.sparring_adapter import (
    OfflineSparringPolicy,
    build_offline_observation,
    from_tactical_env,
)


def semantic_test_map() -> SemanticMap:
    width, height = 28, 15
    healing = np.zeros((height, width), dtype=bool)
    healing[7, 3] = True
    highland = np.zeros((height, width), dtype=bool)
    highland[7, 4] = True
    return SemanticMap(
        width=width,
        height=height,
        hard_blocked=np.zeros((height, width), dtype=bool),
        static_cost=np.zeros((height, width), dtype=np.float32),
        anchors={"hold": (3, 7)},
        red_base=(1, 7),
        blue_base=(26, 7),
        semantic_layers={"healing": healing.astype(np.float32), "central_highland": highland.astype(np.float32)},
        region_masks={"healing_red": healing, "central_highland_1": highland},
        region_kinds={"healing_red": "healing", "central_highland_1": "central_highland"},
        region_centers={"healing_red": (3, 7), "central_highland_1": (4, 7)},
    )


class TacticalSemanticsTest(unittest.TestCase):
    def test_healing_zone_and_highland_defence_apply(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1)
        env.sentry.cell = (3, 7)
        env.sentry.hp = 200.0

        healed = env._apply_semantic_effects()

        self.assertEqual(healed, 40.0)
        self.assertEqual(env.sentry.hp, 240.0)
        env.sentry.cell = (4, 7)
        self.assertEqual(env._unit_defense_bonus(env.sentry), 0.25)

    def test_sentry_building_target_damages_an_actual_outpost(self) -> None:
        env = SentryTacticalEnv(seed=1)
        env.sentry.cell = env._structure_cell("blue", "outpost")

        class AlwaysHit:
            @staticmethod
            def random() -> float:
                return 0.0

        env.rng = AlwaysHit()
        damage = env._sentry_fire(env.BLUE_OUTPOST_TARGET)

        self.assertEqual(damage, 20.0)
        self.assertEqual(env.match.blue.outpost_hp, 1480.0)

    def test_delivered_semantic_map_keeps_geometry_in_declared_world_orientation(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tactical_map = SemanticMap.from_aligned_json(root / "sentry_tactical_rl/assets/semantic_map_aligned.json")

        self.assertIn("healing_2", tactical_map.region_ids_at(tactical_map.region_centers["healing_2"]))
        self.assertEqual(tactical_map.region_owner("healing_2"), "red")
        self.assertEqual(tactical_map.region_owner("healing_1"), "blue")
        self.assertLess(tactical_map.region_centers["healing_2"][1], tactical_map.height // 2)

    def test_offline_adapter_reuses_the_161d_referee_contract(self) -> None:
        env = SentryTacticalEnv(seed=3)
        env.match.time_s = 123.0
        snapshot = from_tactical_env(env)

        observation = build_offline_observation(snapshot, ego_role="sentry", ego_team="blue")

        self.assertEqual(observation.shape, (161,))
        self.assertTrue(np.isfinite(observation).all())
        self.assertAlmostEqual(float(observation[15]), 123.0 / 420.0)
        self.assertAlmostEqual(float(observation[16]), 1.0 - 123.0 / 420.0)

    def test_frozen_blue_sentry_checkpoint_can_act_on_a_tactical_snapshot(self) -> None:
        root = Path(__file__).resolve().parents[1]
        env = SentryTacticalEnv(seed=4)
        policy = OfflineSparringPolicy(
            str(root / "rm_runs/blue_sentry_iql_tactical"),
            semantic_map=env.map,
            team="blue",
            role="sentry",
        )

        command = policy.act(from_tactical_env(env))

        self.assertTrue(env.map.is_free(command.goal_cell))
        self.assertTrue(np.isfinite(command.target_confidence))


if __name__ == "__main__":
    unittest.main()
