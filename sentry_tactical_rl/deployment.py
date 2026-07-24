"""Serialize high-level decisions without controlling the chassis directly.

The output mirrors the JSON action shape already used by the neighbouring
Gazebo project's tactical protocol.  A future ROS2/CDC bridge can consume this
object; this package deliberately does not import ``rclpy`` or publish
``cmd_vel``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .env import SentryTacticalEnv
from .semantic_map import Cell, SemanticMap


@dataclass(frozen=True)
class TacticalDecision:
    goal_index: int
    target_index: int
    fire_mode: int
    skill: str = "go_to"
    ttl_ms: int = 800


_ACTION_TYPE = {
    "hold": "HOLD",
    "go_to": "GO_TO",
    "defend": "DEFEND",
    "retreat": "RETREAT",
    "pressure": "PURSUE",
    "support": "SUPPORT",
    "patrol": "PATROL",
}


def goal_pose_json(cell: Cell, *, cell_size_m: float = 1.0) -> dict[str, Any]:
    """Create the standard ``/navigation/goal_pose`` JSON representation."""
    return {
        "header": {"frame_id": "map"},
        "pose": {
            "position": {"x": (cell[0] + 0.5) * cell_size_m, "y": (cell[1] + 0.5) * cell_size_m, "z": 0.0},
            "orientation": {"z": 0.0, "w": 1.0},
        },
    }


def tactical_action_json(decision: TacticalDecision, semantic_map: SemanticMap) -> dict[str, Any]:
    """Return a schema-constrained tactical command for a radar/CDC bridge."""
    if not 0 <= decision.goal_index < len(semantic_map.anchor_names):
        raise ValueError("goal_index outside configured semantic anchors")
    goal_name = semantic_map.anchor_names[decision.goal_index]
    goal = semantic_map.anchors[goal_name]
    has_target = decision.target_index != SentryTacticalEnv.NONE_TARGET
    constraints = ["AVOID_EXPOSURE", "ALLOW_SEMANTIC_PRIOR"]
    if decision.fire_mode == SentryTacticalEnv.FIRE_ENGAGE:
        constraints.append("ALLOW_FIRE")
    return {
        "valid": True,
        "ttl_ms": int(decision.ttl_ms),
        "type": _ACTION_TYPE.get(decision.skill, "GO_TO"),
        "target": {
            "type": "ENEMY_ROBOT" if has_target else "SEMANTIC_ZONE",
            "roster_index": int(decision.target_index) if has_target else None,
            "semantic_zone": None if has_target else goal_name,
            "goal_map_m": {"x": goal[0] + 0.5, "y": goal[1] + 0.5, "yaw_deg": None},
        },
        "constraints": constraints,
        "fallback": "LOCAL_UTILITY",
        "route_profile": "SAFE",
    }
