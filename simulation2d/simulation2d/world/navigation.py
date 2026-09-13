"""Navigation-backend boundary for high-level tactical decisions.

The demo backend uses :class:`SemanticMap.plan`.  Production should implement
the same ``plan`` return contract by querying the ROS2 navigation stack; the
existing Gazebo project already accepts goals on ``/navigation/goal_pose``.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .semantic_map import Cell, PathPlan, SemanticMap


@dataclass
class NavigationResult:
    reachable: bool
    path_cost: float
    path: list[Cell]
    reason: str = "ok"


class GridNavigationBackend:
    """Reference backend used by the 2D simulator, never by real hardware."""

    def __init__(self, semantic_map: SemanticMap):
        self.map = semantic_map

    def plan(self, start: Cell, goal: Cell, dynamic_cost: np.ndarray) -> NavigationResult:
        plan: PathPlan = self.map.plan(start, goal, dynamic_cost)
        return NavigationResult(
            reachable=plan.reachable,
            path_cost=plan.cost,
            path=plan.cells,
            reason="ok" if plan.reachable else "goal_unreachable",
        )
