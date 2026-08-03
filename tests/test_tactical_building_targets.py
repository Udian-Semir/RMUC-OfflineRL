"""Regression tests for explicit offline outpost/base target labels."""
from __future__ import annotations

import unittest

import numpy as np

from rm_rl.data import schema as S
from rm_rl.data.features import Entity, GameArrays, build_action_raw, target_soft


def entity(*, x: float = 0.0, y: float = 0.0, hp: tuple[float, ...] = (400.0, 400.0),
           yaw: tuple[float, ...] = (0.0, 0.0), ammo17: tuple[float, ...] = (0.0, 0.0),
           ammo42: tuple[float, ...] = (0.0, 0.0)) -> Entity:
    t = len(hp)
    zeros = np.zeros(t, dtype=np.float32)
    return Entity(
        x=np.full(t, x, dtype=np.float32), y=np.full(t, y, dtype=np.float32), z=zeros,
        hp=np.asarray(hp, dtype=np.float32), maxhp=np.full(t, 400.0, dtype=np.float32),
        yaw=np.asarray(yaw, dtype=np.float32), power=zeros, heat17=zeros,
        heat17_max=np.full(t, 260.0, dtype=np.float32), heat42=zeros, heat42_max=zeros,
        ammo17=np.asarray(ammo17, dtype=np.float32), ammo42=np.asarray(ammo42, dtype=np.float32),
        coin_left=zeros, coin_total=zeros, vuln=zeros,
        alive=(np.asarray(hp, dtype=np.float32) > 0).astype(np.float32),
    )


class TacticalBuildingTargetTest(unittest.TestCase):
    def test_outpost_target_aligns_with_same_transition_fire_and_hp_drop(self) -> None:
        # Red hero faces the fixed blue-outpost landmark. At sample 1 it has
        # fired and the blue outpost has lost HP, which labels transition 0.
        yaw = float(np.degrees(np.arctan2(11.12 - 10.0, 16.95 - 10.0)))
        entities = {
            S.robot_id(S.TYPE_HERO, S.CAMP_RED): entity(x=10.0, y=10.0, yaw=(yaw, yaw), ammo42=(0.0, 1.0)),
            S.robot_id(S.TYPE_OUTPOST, S.CAMP_BLUE): entity(hp=(1500.0, 1480.0)),
            S.robot_id(S.TYPE_BASE, S.CAMP_BLUE): entity(hp=(5000.0, 5000.0)),
        }
        game = GameArrays(2, entities)

        action = build_action_raw(game, S.CAMP_RED, S.TYPE_HERO, action_mode="tactical", goal_horizon=1)

        self.assertEqual(action.shape, (1, 12))
        self.assertEqual(float(action[0, 2]), 1.0)
        self.assertEqual(int(action[0, 3:].argmax()), 6)  # six vehicles, then enemy outpost

    def test_stationary_in_range_without_vehicle_gets_weak_building_label(self) -> None:
        # Three 1-Hz samples mean the robot has been stationary for two
        # elapsed seconds.  No shot/HP-drop is present, so this exercises the
        # explicit weak-label fallback rather than the observed-damage label.
        yaw = float(np.degrees(np.arctan2(11.12 - 11.0, 16.95 - 14.0)))
        entities = {
            S.robot_id(S.TYPE_HERO, S.CAMP_RED): entity(
                x=14.0, y=11.0, hp=(400.0, 400.0, 400.0),
                # The weak rule is a tactical target decision; it must not
                # require the current turret heading to already face it.
                yaw=(yaw + 90.0, yaw + 90.0, yaw + 90.0), ammo42=(0.0, 0.0, 0.0)),
            S.robot_id(S.TYPE_OUTPOST, S.CAMP_BLUE): entity(
                hp=(1500.0, 1500.0, 1500.0)),
            S.robot_id(S.TYPE_BASE, S.CAMP_BLUE): entity(
                hp=(5000.0, 5000.0, 5000.0)),
        }
        labels = target_soft(GameArrays(3, entities), S.CAMP_RED, S.TYPE_HERO)
        self.assertEqual(int(labels[2].argmax()), 6)

    def test_nearby_vehicle_suppresses_weak_building_label(self) -> None:
        yaw = float(np.degrees(np.arctan2(11.12 - 11.0, 16.95 - 14.0)))
        entities = {
            S.robot_id(S.TYPE_HERO, S.CAMP_RED): entity(
                x=14.0, y=11.0, hp=(400.0, 400.0, 400.0),
                yaw=(yaw, yaw, yaw), ammo42=(0.0, 0.0, 0.0)),
            S.robot_id(S.TYPE_INFANTRY3, S.CAMP_BLUE): entity(
                x=15.0, y=11.0, hp=(400.0, 400.0, 400.0)),
            S.robot_id(S.TYPE_OUTPOST, S.CAMP_BLUE): entity(
                hp=(1500.0, 1500.0, 1500.0)),
            S.robot_id(S.TYPE_BASE, S.CAMP_BLUE): entity(
                hp=(5000.0, 5000.0, 5000.0)),
        }
        labels = target_soft(GameArrays(3, entities), S.CAMP_RED, S.TYPE_HERO)
        self.assertNotEqual(int(labels[2].argmax()), 6)  # building suppressed


if __name__ == "__main__":
    unittest.main()
