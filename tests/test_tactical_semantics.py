"""Tests that semantic geometry is usable by the tactical rule environment."""
from __future__ import annotations

from pathlib import Path
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from sentry_tactical_rl.env import SentryTacticalEnv
from sentry_tactical_rl.semantic_map import SemanticMap
from sentry_tactical_rl.sparring_adapter import (
    OfflineSparringPolicy,
    build_offline_observation,
    from_tactical_env,
)
from rm_rl.data.features import obs_feature_names


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
    @staticmethod
    def _prime_building_hold(env: SentryTacticalEnv, attacker, kind: str) -> None:
        for opponent in env._team_units("blue" if attacker.team == "red" else "red"):
            if opponent is not attacker:
                opponent.hp = 0.0
        for step in (0, 1):
            env.step_count = step
            assert not env._building_assault_ready(attacker, kind)
        env.step_count = 2

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
        self._prime_building_hold(env, env.sentry, "outpost")
        damage = env._sentry_fire(env.BLUE_OUTPOST_TARGET)

        self.assertEqual(damage, 20.0)
        self.assertEqual(env.match.blue.outpost_hp, 1480.0)

    def test_aggressive_blue_autoaim_fires_when_sparse_offline_gate_is_closed(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1, sparring_fire_policy="aggressive_blue")
        blue = env.enemies[-1]  # blue sentry
        env.sentry.cell = (4, 7)
        blue.cell = (5, 7)

        class AlwaysHit:
            @staticmethod
            def random() -> float:
                return 0.0

        env.rng = AlwaysHit()
        result = {key: 0.0 for key in (
            "red_sentry_damage", "red_outpost_damage", "red_base_damage",
            "blue_outpost_damage", "blue_base_damage",
        )}
        # The frozen policy has selected the sentry but did not predict a
        # sparse next-second shot event.  Auto-aim may execute that intent.
        closed_gate = SimpleNamespace(fire_allowed=False, target_role="sentry")
        env._execute_sparring_fire(blue, closed_gate, result)

        self.assertEqual(blue.shots_fired, 1)
        # (4, 7) is the test map's central-highland cell, so the normal 25%
        # defensive semantic modifier remains active under auto-aim.
        self.assertEqual(result["red_sentry_damage"], 135.0)
        self.assertEqual(env.sentry.hp, env.sentry.max_hp - 135.0)

    def test_engineer_never_shoots_under_aggressive_autoaim(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1, sparring_fire_policy="aggressive_blue")
        engineer = env.enemies[1]
        env.sentry.cell = (4, 7)
        engineer.cell = (5, 7)
        result = {key: 0.0 for key in (
            "red_sentry_damage", "red_outpost_damage", "red_base_damage",
            "blue_outpost_damage", "blue_base_damage",
        )}
        env._execute_sparring_fire(engineer, SimpleNamespace(fire_allowed=False, target_role=None), result)

        self.assertEqual(engineer.shots_fired, 0)
        self.assertEqual(result["red_sentry_damage"], 0.0)

    def test_building_damage_requires_fire_gate_in_addition_to_a_nearby_goal(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1, sparring_fire_policy="aggressive_blue")
        blue = env.enemies[-1]
        blue.cell = (3, 7)
        red_outpost = env._structure_cell("red", "outpost")
        result = {key: 0.0 for key in (
            "red_sentry_damage", "red_outpost_damage", "red_base_damage",
            "blue_outpost_damage", "blue_base_damage",
        )}
        closed_gate = SimpleNamespace(fire_allowed=False, target_role=None, goal_cell=red_outpost)
        env._execute_sparring_fire(blue, closed_gate, result)

        self.assertEqual(blue.shots_fired, 0)
        self.assertEqual(result["red_outpost_damage"], 0.0)

        open_gate = SimpleNamespace(fire_allowed=True, target_role=None, goal_cell=red_outpost)
        self._prime_building_hold(env, blue, "outpost")
        env._execute_sparring_fire(blue, open_gate, result)

        self.assertEqual(blue.shots_fired, 1)
        self.assertEqual(result["red_outpost_damage"], 20.0)

    def test_explicit_offline_outpost_target_uses_effective_building_damage(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1, sparring_fire_policy="intent_dps")
        infantry = env.enemies[2]
        infantry.cell = (3, 7)
        result = {key: 0.0 for key in (
            "red_sentry_damage", "red_outpost_damage", "red_base_damage",
            "blue_outpost_damage", "blue_base_damage",
        )}

        # New 12-D checkpoints emit 前哨站 directly. It is sufficient building
        # intent under intent_dps once the vehicle has completed the required
        # clear two-second building hold.
        command = SimpleNamespace(fire_allowed=False, target_role="前哨站", goal_cell=(3, 7))
        self._prime_building_hold(env, infantry, "outpost")
        env._execute_sparring_fire(infantry, command, result)

        self.assertEqual(result["red_outpost_damage"], 200.0)
        self.assertEqual(env.match.red.outpost_hp, 1300.0)

    def test_selected_mobile_target_overrides_unrelated_goal_with_firing_position(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1)
        attacker, target = env.allies[2], env.enemies[2]
        attacker.cell, target.cell = (3, 7), (20, 7)
        command = SimpleNamespace(goal_cell=(3, 7), target_role="步兵3")

        goal = env._sparring_execution_goal(attacker, command)

        self.assertNotEqual(goal, command.goal_cell)
        self.assertLessEqual(env._distance(goal, target.cell), attacker.attack_range)
        self.assertTrue(env.map.line_of_sight(goal, target.cell))

    def test_mid_hp_sentry_still_couples_target_to_firing_position(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1)
        target = env.enemies[2]
        env.sentry.hp = 156.0
        env.sentry.cell, target.cell = (3, 7), (20, 7)

        goal = env._sentry_execution_goal(env.sentry.cell, 2)

        self.assertNotEqual(goal, env.sentry.cell)
        self.assertGreaterEqual(env._distance(goal, target.cell), 2.0)
        self.assertLessEqual(env._distance(goal, target.cell), 3.0)
        self.assertTrue(env.map.line_of_sight(goal, target.cell))

    def test_sentry_recovery_starts_only_below_twenty_percent(self) -> None:
        at_threshold = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1)
        at_threshold.sentry.hp = at_threshold.sentry.max_hp * 0.20
        action = at_threshold._scheduled_sentry_action(0, 2, at_threshold.FIRE_ENGAGE)
        self.assertEqual(action, (0, 2, at_threshold.FIRE_ENGAGE))

        below_threshold = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1)
        below_threshold.sentry.hp = below_threshold.sentry.max_hp * 0.20 - 0.1
        goal, target, fire = below_threshold._scheduled_sentry_action(
            0, 2, below_threshold.FIRE_ENGAGE)

        self.assertEqual(goal, below_threshold._sentry_supply_goal_index())
        self.assertEqual(target, below_threshold.NONE_TARGET)
        self.assertEqual(fire, below_threshold.FIRE_HOLD)

    def test_sentry_latches_static_goal_until_it_is_reached(self) -> None:
        tactical_map = SemanticMap.demo()
        env = SentryTacticalEnv(
            semantic_map=tactical_map, seed=1, goal_hold_seconds=1.0,
            goal_reach_tolerance_m=0.35,
        )
        env.sentry.invulnerable_until_s = 999.0
        goal_indices = sorted(
            range(env.n_goals),
            key=lambda index: env._distance(
                env.sentry.cell, env.map.anchors[env.anchor_names[index]]),
            reverse=True,
        )
        first_goal, second_goal = goal_indices[:2]

        _, _, _, first = env.step((first_goal, env.NONE_TARGET, env.FIRE_HOLD))
        latched_cell = first["execution_goal_cell"]
        _, _, _, blocked = env.step((second_goal, env.NONE_TARGET, env.FIRE_HOLD))

        self.assertTrue(blocked["goal_switch_blocked"])
        self.assertFalse(blocked["goal_target_replan"])
        self.assertEqual(blocked["execution_goal_cell"], latched_cell)

        env.sentry.cell = latched_cell
        _, _, _, arrival = env.step((second_goal, env.NONE_TARGET, env.FIRE_HOLD))
        self.assertTrue(arrival["goal_reached"])
        _, _, _, released = env.step((second_goal, env.NONE_TARGET, env.FIRE_HOLD))

        self.assertFalse(released["goal_switch_blocked"])
        self.assertEqual(released["executed_goal_idx"], second_goal)

    def test_sentry_replans_standoff_after_reaching_current_route(self) -> None:
        env = SentryTacticalEnv(
            semantic_map=SemanticMap.demo(), seed=1, goal_hold_seconds=1.0,
            goal_reach_tolerance_m=0.35,
        )
        env.sentry.invulnerable_until_s = 999.0
        goal_idx = max(
            range(env.n_goals),
            key=lambda index: env._distance(
                env.sentry.cell, env.map.anchors[env.anchor_names[index]]),
        )
        target = env.enemies[2]
        target.cell = (20, 7)

        _, _, _, first = env.step((goal_idx, 2, env.FIRE_HOLD))
        first_goal = first["execution_goal_cell"]
        target.cell = (16, 11)
        _, _, _, deferred = env.step((goal_idx, 2, env.FIRE_HOLD))

        self.assertFalse(deferred["goal_target_replan"])
        self.assertTrue(deferred["goal_target_replan_deferred"])
        self.assertTrue(deferred["goal_switch_blocked"])
        self.assertFalse(deferred["concrete_goal_changed"])
        self.assertEqual(deferred["execution_goal_cell"], first_goal)

        env.sentry.cell = first_goal
        env.active_goal_reached = True
        target.cell = (16, 11)
        _, _, _, replanned = env.step((goal_idx, 2, env.FIRE_HOLD))

        self.assertTrue(replanned["goal_target_replan"])
        self.assertFalse(replanned["goal_target_replan_deferred"])
        self.assertTrue(replanned["concrete_goal_changed"])
        self.assertFalse(replanned["goal_switch"])
        self.assertNotEqual(replanned["execution_goal_cell"], first_goal)
        self.assertLessEqual(
            env._distance(replanned["execution_goal_cell"], target.cell),
            env.sentry.attack_range,
        )


    def test_selected_protected_base_keeps_learned_goal(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1)
        attacker = env.enemies[2]
        command = SimpleNamespace(goal_cell=(18, 7), target_role="基地")

        self.assertEqual(env._sparring_execution_goal(attacker, command), command.goal_cell)

    def test_hero_42mm_damage_has_four_second_cadence(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1)
        hero, target = env.allies[0], env.enemies[0]
        hero.cell, target.cell = (5, 7), (6, 7)

        first = env._fire_unit_at_robot(hero, target)
        second = env._fire_unit_at_robot(hero, target)
        env.match.advance(4.0)
        third = env._fire_unit_at_robot(hero, target)

        self.assertEqual((first, second, third), (200.0, 0.0, 200.0))

    def test_regular_vehicle_damage_uses_the_two_effective_dps_rings(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1)
        attacker, target = env.allies[2], env.enemies[2]
        attacker.cell, target.cell = (10, 10), (12, 10)

        close_damage = env._fire_unit_at_robot(attacker, target)
        env.match.advance(1.0)
        attacker.cell, target.cell = (10, 10), (14, 10)
        mid_damage = env._fire_unit_at_robot(attacker, target)

        self.assertEqual((close_damage, mid_damage), (140.0, 70.0))
        self.assertEqual([event["band"] for event in env._combat_events], ["close_0_3m", "mid_3_5m"])

    def test_sentry_vehicle_damage_uses_180_and_100_dps_rings(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1)
        target = env.enemies[2]
        env.sentry.cell, target.cell = (10, 10), (12, 10)

        close_damage = env._sentry_fire(2)
        env.match.advance(1.0)
        env.sentry.cell, target.cell = (10, 10), (14, 10)
        mid_damage = env._sentry_fire(2)

        self.assertEqual((close_damage, mid_damage), (180.0, 100.0))

    def test_one_minute_phase_forces_living_blue_outpost_target(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1)
        env.match.time_s = 60.0

        _, target, fire = env._scheduled_sentry_action(0, env.NONE_TARGET, env.FIRE_HOLD)

        self.assertEqual(target, env.BLUE_OUTPOST_TARGET)
        self.assertEqual(fire, env.FIRE_ENGAGE)
        fixed_goal = env._sentry_outpost_attack_goal()
        self.assertAlmostEqual(env._distance(fixed_goal, env.map.blue_outpost), 7.0, delta=0.25)
        self.assertEqual(env._sentry_execution_goal((3, 7), env.BLUE_OUTPOST_TARGET), fixed_goal)

    def test_designated_seven_meter_lane_can_hit_elevated_outpost(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1)
        fixed_goal = env._sentry_outpost_attack_goal()
        env.sentry.cell = fixed_goal
        assert env.map.raw_hard_blocked is not None
        env.map.raw_hard_blocked[7, 22] = True
        self.assertFalse(env.map.line_of_sight(env.sentry.cell, env.map.blue_outpost))
        self._prime_building_hold(env, env.sentry, "outpost")

        damage = env._sentry_fire(env.BLUE_OUTPOST_TARGET)

        self.assertEqual(damage, 20.0)
        env.match.advance(1.0)
        env.sentry.cell = (fixed_goal[0], fixed_goal[1] - 5)
        self.assertEqual(env._sentry_fire(env.BLUE_OUTPOST_TARGET), 0.0)

    def test_one_minute_outpost_order_interrupts_for_lethal_threat(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1)
        env.match.time_s = 60.0
        env.sentry.cell = (10, 7)
        env.sentry.hp = 100.0
        attacker = env.enemies[2]
        attacker.cell = (12, 7)

        _, target, fire = env._scheduled_sentry_action(0, env.BLUE_OUTPOST_TARGET, env.FIRE_ENGAGE)

        self.assertEqual(target, 2)
        self.assertEqual(fire, env.FIRE_ENGAGE)
        self.assertIs(env._imminent_sentry_threat(), attacker)

    def test_regular_vehicle_effective_dps_is_symmetric_and_buildings_use_200_per_second(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1)
        blue_infantry, red_infantry = env.enemies[2], env.allies[2]
        blue_infantry.cell, red_infantry.cell = (12, 10), (10, 10)

        self.assertEqual(env._fire_unit_at_robot(blue_infantry, red_infantry), 140.0)
        env.match.advance(1.0)
        blue_infantry.cell = env._structure_cell("red", "outpost")
        result = {key: 0.0 for key in (
            "red_sentry_damage", "red_outpost_damage", "red_base_damage",
            "blue_outpost_damage", "blue_base_damage",
        )}
        self._prime_building_hold(env, blue_infantry, "outpost")
        env._fire_unit_at_structure(blue_infantry, result)

        self.assertEqual(result["red_outpost_damage"], 200.0)

    def test_aerial_unit_cannot_be_damaged_by_ground_fire(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1)
        env.sentry.cell = (3, 7)
        aerial = env.enemies[4]
        aerial.cell = (4, 7)

        self.assertEqual(env._fire_unit_at_robot(env.sentry, aerial), 0.0)
        self.assertEqual(aerial.hp, aerial.max_hp)

    def test_aerial_outpost_intent_uses_stable_airborne_standoff(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1)
        aerial = env.allies[4]
        command = SimpleNamespace(goal_cell=(1, 1), target_role="前哨站")

        goal = env._sparring_execution_goal(aerial, command)

        self.assertNotEqual(goal, command.goal_cell)
        self.assertGreaterEqual(env._distance(goal, env.map.blue_outpost), 2.0)
        self.assertLessEqual(env._distance(goal, env.map.blue_outpost), 3.0)

    def test_aerial_explicit_outpost_intent_fires_above_ground_occlusion(self) -> None:
        env = SentryTacticalEnv(
            semantic_map=semantic_test_map(), seed=1, sparring_fire_policy="intent_dps")
        aerial = env.allies[4]
        aerial.cell = env._aerial_structure_standoff_goal(aerial, env.map.blue_outpost)
        assert env.map.raw_hard_blocked is not None
        env.map.raw_hard_blocked[7, 21] = True
        self.assertFalse(env.map.line_of_sight(aerial.cell, env.map.blue_outpost))
        self._prime_building_hold(env, aerial, "outpost")
        command = SimpleNamespace(
            goal_cell=(1, 1), target_role="前哨站", fire_allowed=False)
        result = {key: 0.0 for key in (
            "red_sentry_damage", "red_outpost_damage", "red_base_damage",
            "blue_outpost_damage", "blue_base_damage",
        )}

        env._execute_sparring_fire(aerial, command, result)

        self.assertEqual(result["blue_outpost_damage"], 200.0)

    def test_ground_robot_respawns_at_own_supply_after_reader_finishes(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=1)
        infantry = env.enemies[2]
        infantry.hp = 0.0
        infantry.ammo = 37
        infantry.respawn_remaining_s = 1.0
        env._advance_respawns()

        self.assertTrue(infantry.alive)
        self.assertAlmostEqual(infantry.hp, infantry.max_hp * 0.10)
        self.assertEqual(infantry.ammo, 37)
        self.assertEqual(infantry.cell, env.map.blue_base)

    def test_scripted_backend_advances_ground_respawn_reader(self) -> None:
        env = SentryTacticalEnv(
            semantic_map=SemanticMap.demo(), horizon=4, seed=17,
            sparring_backend="scripted",
        )
        observation = env.reset(seed=17)
        unit = env.enemies[2]
        unit.hp = 0.0
        unit.respawn_remaining_s = 1.0
        goal = int(np.flatnonzero(observation["goal_mask"])[0])

        env.step((goal, env.NONE_TARGET, env.FIRE_HOLD))

        self.assertTrue(unit.alive)
        self.assertEqual(unit.hp, unit.max_hp * 0.10)

    def test_blue_sparring_profile_is_fixed_for_an_episode_and_reset_selects_from_pool(self) -> None:
        env = SentryTacticalEnv(
            semantic_map=semantic_test_map(),
            seed=21,
            sparring_backend="offline",
            sparring_fire_policy="aggressive_blue",
            blue_sparring_profiles=("pressure", "measured"),
        )
        first_profile = env.blue_sparring_profile
        first_probability = env.blue_autoaim_probability
        self.assertIn(first_profile, {"pressure", "measured"})

        observation = env.observe()
        action = (int(np.flatnonzero(observation["goal_mask"])[0]), env.NONE_TARGET, env.FIRE_HOLD)
        _, _, _, info = env.step(action)
        self.assertEqual(info["blue_sparring_profile"], first_profile)
        self.assertEqual(info["blue_autoaim_probability"], first_probability)
        env.reset()
        self.assertIn(env.blue_sparring_profile, {"pressure", "measured"})

    def test_blue_style_profile_selects_its_own_frozen_policy(self) -> None:
        root = Path(__file__).resolve().parents[1]
        style_dirs = {
            profile: {
                "hero": f"sparring_style_pool/checkpoints/{profile}/hero",
                "engineer": f"sparring_style_pool/checkpoints/{profile}/engineer",
                "infantry3": f"sparring_style_pool/checkpoints/{profile}/infantry",
                "infantry4": f"sparring_style_pool/checkpoints/{profile}/infantry",
                "aerial": f"sparring_style_pool/checkpoints/{profile}/aerial",
                "sentry": f"sparring_style_pool/checkpoints/{profile}/sentry",
            }
            for profile in ("aggressive", "measured")
        }
        env = SentryTacticalEnv(
            semantic_map=semantic_test_map(),
            seed=21,
            sparring_backend="offline",
            blue_sparring_profiles=("aggressive", "measured"),
            blue_sparring_style_run_dirs=style_dirs,
        )
        blue_sentry = env.enemies[-1]

        env.blue_sparring_profile = "aggressive"
        aggressive = env._sparring_policy_for(blue_sentry)
        env.blue_sparring_profile = "measured"
        measured = env._sparring_policy_for(blue_sentry)

        self.assertIsNot(aggressive, measured)
        self.assertEqual(
            env._blue_style_policy_keys[("aggressive", "sentry")][1],
            str((root / style_dirs["aggressive"]["sentry"]).resolve()),
        )
        self.assertEqual(
            env._blue_style_policy_keys[("measured", "sentry")][1],
            str((root / style_dirs["measured"]["sentry"]).resolve()),
        )

    def test_delivered_semantic_map_keeps_geometry_in_declared_world_orientation(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tactical_map = SemanticMap.from_aligned_json(root / "sentry_tactical_rl/assets/semantic_map_aligned.json")

        self.assertIn("healing_2", tactical_map.region_ids_at(tactical_map.region_centers["healing_2"]))
        self.assertEqual(tactical_map.region_owner("healing_2"), "red")
        self.assertEqual(tactical_map.region_owner("healing_1"), "blue")
        self.assertLess(tactical_map.region_centers["healing_2"][1], tactical_map.height // 2)

    def test_aligned_map_spawns_every_unit_on_free_cell(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tactical_map = SemanticMap.from_aligned_json(
            root / "sentry_tactical_rl/assets/semantic_map_aligned.json",
            obstacle_path=root / "sentry_tactical_rl/assets/blackwhite_map.png",
        )
        env = SentryTacticalEnv(semantic_map=tactical_map, seed=2)

        self.assertAlmostEqual(tactical_map.resolution_m, 0.1)
        self.assertTrue(all(tactical_map.is_free(unit.cell)
                            for unit in (env.sentry, *env.allies, *env.enemies)
                            if unit.role != "aerial"))
        self.assertEqual(
            tactical_map.cell_to_meters(env.allies[-1].cell),
            tactical_map.cell_to_meters(tactical_map.meters_to_cell((1.41, 13.39))),
        )

    def test_delivered_map_astar_reaches_all_anchors_without_crossing_obstacles(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tactical_map = SemanticMap.from_aligned_json(root / "sentry_tactical_rl/assets/semantic_map_aligned.json")
        sources = {
            "red_base": tactical_map.nearest_free(tactical_map.red_base),
            "red_outpost": tactical_map.nearest_free(tactical_map.red_outpost),
            "blue_outpost": tactical_map.nearest_free(tactical_map.blue_outpost),
            "blue_base": tactical_map.nearest_free(tactical_map.blue_base),
        }
        self.assertTrue(all(cell is not None for cell in sources.values()))
        anchors = {name: tactical_map.nearest_free(cell) for name, cell in tactical_map.anchors.items()}
        self.assertTrue(all(cell is not None for cell in anchors.values()))

        # The tactical grid must retain the three strategic cross-field routes
        # and every selectable semantic anchor must be reachable from each end.
        pairs = [
            (sources["red_base"], sources["red_outpost"]),
            (sources["red_outpost"], sources["blue_outpost"]),
            (sources["blue_outpost"], sources["blue_base"]),
        ]
        pairs.extend((source, anchor) for source in sources.values() for anchor in anchors.values())
        for start, goal in pairs:
            assert start is not None and goal is not None
            plan = tactical_map.plan(start, goal)
            self.assertTrue(plan.reachable, msg=f"unreachable: {start} -> {goal}")
            self.assertEqual(plan.cells[0], start)
            self.assertEqual(plan.cells[-1], goal)
            for current, nxt in zip(plan.cells, plan.cells[1:]):
                self.assertTrue(tactical_map.is_free(nxt))
                self.assertEqual(abs(current[0] - nxt[0]) + abs(current[1] - nxt[1]), 1)

    def test_delivered_map_goals_respect_gazebo_clearance(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tactical_map = SemanticMap.from_aligned_json(
            root / "sentry_tactical_rl/assets/semantic_map_aligned.json",
            obstacle_path=root / "sentry_tactical_rl/assets/blackwhite_map.png",
        )
        assert tactical_map.raw_hard_blocked is not None
        distance_m = cv2.distanceTransform(
            (~tactical_map.raw_hard_blocked).astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        ) * tactical_map.resolution_m

        for name, (x, y) in tactical_map.anchors.items():
            self.assertGreaterEqual(distance_m[y, x] + 1e-6, 0.35, msg=name)
        for name in ("tunnel_gain_1", "tunnel_gain_2", "tunnel_gain_3", "tunnel_gain_4"):
            self.assertTrue(tactical_map.plan(tactical_map.red_base, tactical_map.anchors[name]).reachable)

    def test_delivered_map_uses_exported_inflated_astar_png(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tactical_map = SemanticMap.from_aligned_json(
            root / "sentry_tactical_rl/assets/semantic_map_aligned.json",
            obstacle_path=root / "sentry_tactical_rl/assets/blackwhite_map.png",
        )
        image = cv2.imread(
            str(root / "sentry_tactical_rl/assets/blackwhite_astar_inflated_0p35m.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        self.assertIsNotNone(image)
        exported_hard = np.flipud(image <= 5)

        self.assertTrue(np.array_equal(tactical_map.hard_blocked, exported_hard))

    def test_non_sentry_companions_use_uninflated_physical_map(self) -> None:
        root = Path(__file__).resolve().parents[1]
        tactical_map = SemanticMap.from_aligned_json(
            root / "sentry_tactical_rl/assets/semantic_map_aligned.json",
            obstacle_path=root / "sentry_tactical_rl/assets/blackwhite_map.png",
        )
        env = SentryTacticalEnv(semantic_map=tactical_map, seed=2)
        inflation_only = np.argwhere(env.map.hard_blocked & ~env.companion_map.hard_blocked)
        self.assertGreater(len(inflation_only), 0)
        y, x = inflation_only[0]

        self.assertFalse(env.map.is_free((int(x), int(y))))
        self.assertTrue(env.companion_map.is_free((int(x), int(y))))

    def test_offline_adapter_reuses_the_161d_referee_contract(self) -> None:
        env = SentryTacticalEnv(seed=3)
        env.match.time_s = 123.0
        snapshot = from_tactical_env(env)

        observation = build_offline_observation(snapshot, ego_role="sentry", ego_team="blue")

        self.assertEqual(observation.shape, (161,))
        self.assertTrue(np.isfinite(observation).all())
        self.assertAlmostEqual(float(observation[15]), 123.0 / 420.0)
        self.assertAlmostEqual(float(observation[16]), 1.0 - 123.0 / 420.0)

    def test_offline_adapter_keeps_hero_weapon_telemetry_in_its_own_fields(self) -> None:
        env = SentryTacticalEnv(seed=3)
        observation = build_offline_observation(from_tactical_env(env), ego_role="sentry", ego_team="blue")
        names = obs_feature_names("哨兵")

        self.assertAlmostEqual(float(observation[names.index("enemy[英雄].heat17max")]), 50.0 / 260.0)
        self.assertAlmostEqual(float(observation[names.index("enemy[英雄].heat42max")]), 140.0 / 240.0)

    def test_offline_adapter_uses_neutral_team_prior_when_school_history_is_unknown(self) -> None:
        env = SentryTacticalEnv(seed=3)
        observation = build_offline_observation(from_tactical_env(env), ego_role="sentry", ego_team="blue")
        names = obs_feature_names("哨兵")

        self.assertEqual(
            observation[names.index("team.own_winrate"):names.index("team.opp_durability") + 1].tolist(),
            [0.5, 1.0, 1.0, 0.5, 1.0, 1.0],
        )

    def test_external_sentry_state_skips_ppo_motion_during_hybrid_replay(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=3)
        env.set_external_sentry_state(time_s=12.0, x_m=9.4, y_m=8.2, hp=321.0, max_hp=400.0, yaw_deg=37.0)
        expected_cell = env.sentry.cell

        _, _, _, info = env.step((0, env.NONE_TARGET, env.FIRE_HOLD))

        self.assertTrue(info["sentry_external"])
        self.assertEqual(info["goal_name"], "database_recorded")
        self.assertEqual(env.sentry.cell, expected_cell)
        self.assertEqual(env.sentry.hp, 321.0)

    def test_external_sentry_pose_releases_goal_when_real_robot_arrives(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=3)
        env.active_goal_cell = (10, 8)
        env.active_goal_idx = 0
        env.active_goal_reached = False

        goal_x, goal_y = env.map.cell_to_meters(env.active_goal_cell)
        env.set_external_sentry_state(
            time_s=12.0, x_m=goal_x, y_m=goal_y,
            hp=env.sentry.hp, max_hp=env.sentry.max_hp,
        )

        self.assertTrue(env.active_goal_reached)

    def test_external_sentry_can_keep_interactive_damage_feedback(self) -> None:
        env = SentryTacticalEnv(semantic_map=semantic_test_map(), seed=3)
        target = env.enemies[2]
        env.sentry.cell = (10, 8)
        target.cell = (12, 8)
        x_m, y_m = env.map.cell_to_meters(env.sentry.cell)
        before = target.hp
        env.set_external_sentry_state(
            time_s=12.0, x_m=x_m, y_m=y_m,
            hp=env.sentry.hp, max_hp=env.sentry.max_hp,
            simulate_fire=True,
        )

        _, _, _, info = env.step((0, 2, env.FIRE_ENGAGE))

        self.assertGreater(info["damage_dealt"], 0.0)
        self.assertLess(target.hp, before)

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

    def test_full_roster_offline_sparring_executes_five_second_subgoals(self) -> None:
        env = SentryTacticalEnv(horizon=8, seed=19, sparring_backend="offline")
        expected_roles = {"hero", "engineer", "infantry3", "infantry4", "aerial", "sentry"}

        self.assertEqual({unit.role for unit in (env.sentry, *env.allies)}, expected_roles)
        self.assertEqual({unit.role for unit in env.enemies}, expected_roles)
        self.assertEqual(len(env.allies), 5)
        self.assertEqual(len(env.enemies), 6)
        self.assertEqual(env.n_targets, 9)  # six robots, outpost, base, none

        snapshot = from_tactical_env(env)
        for unit in (*env.allies, *env.enemies):
            observation = build_offline_observation(snapshot, ego_role=unit.role, ego_team=unit.team)
            self.assertEqual(observation.shape, (161,))

        obs = env.observe()
        start_cells = {unit.unit_id: unit.cell for unit in (*env.allies, *env.enemies)}
        action = (int(np.flatnonzero(obs["goal_mask"])[0]), env.NONE_TARGET, env.FIRE_HOLD)
        _, _, _, info = env.step(action)

        self.assertEqual(info["sparring_commands"], 11)
        self.assertEqual(len(env._sparring_commands), 11)
        self.assertTrue(all(env.map.is_free(command.goal_cell) for command in env._sparring_commands.values()))
        self.assertGreater(sum(unit.cell != start_cells[unit.unit_id] for unit in (*env.allies, *env.enemies)), 0)


if __name__ == "__main__":
    unittest.main()
