"""Regression tests for the deterministic RMUC 2026 building rules."""
from __future__ import annotations

import unittest

from sentry_tactical_rl.match_rules import MatchState


class MatchRulesTest(unittest.TestCase):
    def test_base_is_invulnerable_while_outpost_is_alive(self) -> None:
        state = MatchState()

        result = state.apply_base_damage("red", "blue", 200.0)

        self.assertEqual(result.applied, 0.0)
        self.assertEqual(result.blocked_reason, "base_protected_by_outpost")
        self.assertEqual(state.blue.base_hp, 5000.0)
        self.assertEqual(state.blue.base_shield, 150.0)

    def test_outpost_destruction_makes_base_damage_legal_and_consumes_shield_first(self) -> None:
        state = MatchState()

        outpost = state.apply_outpost_damage("red", "blue", 1500.0)
        base = state.apply_base_damage("red", "blue", 200.0)

        self.assertEqual(outpost.hp_damage, 1500.0)
        self.assertTrue(state.blue.outpost_destroyed_ever)
        self.assertEqual(base.shield_damage, 150.0)
        self.assertEqual(base.hp_damage, 50.0)
        self.assertEqual(state.blue.base_hp, 4950.0)
        self.assertEqual(state.red.attack_damage, 1700.0)

    def test_rebuild_opportunity_is_accumulated_and_closes_at_five_minutes(self) -> None:
        state = MatchState()
        state.apply_outpost_damage("blue", "red", 1500.0)
        state.apply_base_damage("blue", "red", 1150.0)

        self.assertTrue(state.can_rebuild_outpost("red"))
        self.assertTrue(state.rebuild_outpost("red"))
        self.assertEqual(state.red.outpost_hp, 750.0)
        state.apply_outpost_damage("blue", "red", 750.0)
        state.advance(300.0)

        self.assertFalse(state.can_rebuild_outpost("red"))
        self.assertFalse(state.rebuild_outpost("red"))

    def test_official_timeout_tie_break_order(self) -> None:
        # Outpost HP decides only when neither side has ever lost an outpost.
        state = MatchState()
        state.red.outpost_hp = 1400.0
        state.blue.outpost_hp = 1300.0
        state.advance(420.0)
        self.assertEqual(state.winner(red_total_hp=500.0, blue_total_hp=900.0), "red")

        # Once only one outpost was ever destroyed, that history outranks its
        # rebuilt HP and all later attack/robot-health comparisons.
        state = MatchState()
        state.apply_outpost_damage("blue", "red", 1500.0)
        state.apply_base_damage("blue", "red", 1150.0)
        self.assertTrue(state.rebuild_outpost("red"))
        state.red.attack_damage = 9999.0
        state.advance(420.0)
        self.assertEqual(state.winner(red_total_hp=999.0, blue_total_hp=1.0), "blue")

        # With matching building state, total attack damage then total robot HP
        # decide in that order.
        state = MatchState()
        state.red.attack_damage = 101.0
        state.blue.attack_damage = 100.0
        state.advance(420.0)
        self.assertEqual(state.winner(red_total_hp=1.0, blue_total_hp=999.0), "red")


if __name__ == "__main__":
    unittest.main()
