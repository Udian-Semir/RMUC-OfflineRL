"""Semantic grid map and lightweight path-cost tools for the tactical demo.

This is deliberately independent from ROS.  Its public ``plan`` contract is
the same information the real policy will receive from the navigation stack:
whether a goal is reachable, its travel cost, and a safe grid path.  The
synthetic map in :meth:`SemanticMap.demo` must later be replaced by the team's
surveyed RMUC semantic map.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


Cell = tuple[int, int]  # (x, y), in one-metre tactical cells


@dataclass
class PathPlan:
    reachable: bool
    cost: float
    cells: list[Cell]


@dataclass
class SemanticMap:
    """Static tactical layers and named, policy-selectable goal anchors."""

    width: int
    height: int
    hard_blocked: np.ndarray  # [H, W], true means this sentry may never enter
    static_cost: np.ndarray  # [H, W], non-negative traversal preference
    anchors: dict[str, Cell] = field(default_factory=dict)
    red_outpost: Cell = (5, 7)
    blue_outpost: Cell = (22, 7)
    red_base: Cell = (1, 7)
    blue_base: Cell = (26, 7)

    def __post_init__(self) -> None:
        shape = (self.height, self.width)
        if self.hard_blocked.shape != shape or self.static_cost.shape != shape:
            raise ValueError(f"map layers must have shape {shape}")
        if np.any(self.static_cost < 0):
            raise ValueError("static_cost must be non-negative")
        for name, cell in self.anchors.items():
            if not self.is_free(cell):
                raise ValueError(f"anchor {name!r} is not traversable: {cell}")

    @classmethod
    def demo(cls) -> "SemanticMap":
        """Build a synthetic 28 m x 15 m map for smoke tests and PPO demos.

        It contains central hard obstacles plus soft-risk lanes.  It is *not*
        a claim about the official field layout.
        """
        width, height = 28, 15
        hard = np.zeros((height, width), dtype=bool)
        # Two central blocks force the planner to choose upper/lower lanes.
        hard[4:10, 12:14] = True
        hard[4:10, 15:17] = True
        # Demonstrate a chassis-specific forbidden slope / narrow passage.
        hard[11:14, 8:11] = True

        cost = np.zeros((height, width), dtype=np.float32)
        cost[0:2, :] = 0.35
        cost[-2:, :] = 0.35
        cost[:, 13:15] += 0.15
        anchors = {
            "red_outpost_cover_w": (4, 4),
            "red_outpost_cover_e": (7, 10),
            "red_supply": (3, 12),
            "mid_upper_cover": (10, 2),
            "mid_lower_cover": (11, 12),
            "mid_upper_cross": (18, 2),
            "mid_lower_cross": (18, 12),
            "blue_outpost_cover_w": (20, 4),
            "blue_outpost_cover_e": (23, 10),
            "blue_supply": (24, 2),
            "safe_retreat": (3, 7),
            "blue_pressure": (21, 7),
        }
        return cls(width, height, hard, cost, anchors)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SemanticMap":
        """Load the deliberately small semantic-map interchange format.

        The format is documented by ``configs/arena_example.yaml``.  This is
        the intended replacement path for the synthetic demo map; a ROS map
        adapter can generate the same YAML after terrain calibration.
        """
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        width, height = int(raw["width"]), int(raw["height"])
        hard = np.zeros((height, width), dtype=bool)
        for x, y in raw.get("hard_blocked", []):
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"hard-blocked cell outside map: {(x, y)}")
            hard[y, x] = True
        cost = np.zeros((height, width), dtype=np.float32)
        for item in raw.get("static_cost", []):
            x, y = item["cell"]
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"cost cell outside map: {(x, y)}")
            cost[y, x] = float(item["cost"])
        anchors = {name: tuple(value) for name, value in raw["anchors"].items()}
        objectives = raw.get("objectives", {})
        return cls(
            width=width,
            height=height,
            hard_blocked=hard,
            static_cost=cost,
            anchors=anchors,
            red_outpost=tuple(objectives.get("red_outpost", (5, 7))),
            blue_outpost=tuple(objectives.get("blue_outpost", (22, 7))),
            red_base=tuple(objectives.get("red_base", (1, 7))),
            blue_base=tuple(objectives.get("blue_base", (26, 7))),
        )

    @property
    def anchor_names(self) -> tuple[str, ...]:
        return tuple(self.anchors)

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, cell: Cell) -> bool:
        x, y = cell
        return self.in_bounds(cell) and not bool(self.hard_blocked[y, x])

    def clamp_cell(self, cell: Cell) -> Cell:
        return (int(np.clip(cell[0], 0, self.width - 1)),
                int(np.clip(cell[1], 0, self.height - 1)))

    def neighbours(self, cell: Cell) -> Iterable[Cell]:
        x, y = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if self.is_free(nxt):
                yield nxt

    def plan(self, start: Cell, goal: Cell, dynamic_cost: np.ndarray | None = None) -> PathPlan:
        """A* over hard constraints and static/dynamic soft costs."""
        if not self.is_free(start) or not self.is_free(goal):
            return PathPlan(False, float("inf"), [])
        if dynamic_cost is None:
            dynamic_cost = np.zeros_like(self.static_cost)
        if dynamic_cost.shape != self.static_cost.shape:
            raise ValueError("dynamic_cost shape does not match map")

        total_cost = self.static_cost + np.maximum(dynamic_cost, 0.0)
        queue: list[tuple[float, float, Cell]] = []
        heapq.heappush(queue, (0.0, 0.0, start))
        came_from: dict[Cell, Cell | None] = {start: None}
        g_score: dict[Cell, float] = {start: 0.0}

        while queue:
            _, current_g, current = heapq.heappop(queue)
            if current_g != g_score.get(current):
                continue
            if current == goal:
                cells: list[Cell] = []
                node: Cell | None = current
                while node is not None:
                    cells.append(node)
                    node = came_from[node]
                cells.reverse()
                return PathPlan(True, current_g, cells)
            for nxt in self.neighbours(current):
                nx, ny = nxt
                candidate = current_g + 1.0 + float(total_cost[ny, nx])
                if candidate >= g_score.get(nxt, float("inf")):
                    continue
                g_score[nxt] = candidate
                came_from[nxt] = current
                heuristic = abs(goal[0] - nx) + abs(goal[1] - ny)
                heapq.heappush(queue, (candidate + heuristic, candidate, nxt))
        return PathPlan(False, float("inf"), [])

    def line_of_sight(self, a: Cell, b: Cell) -> bool:
        """Conservative Bresenham visibility; blocked interior cells occlude."""
        x0, y0 = a
        x1, y1 = b
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while (x, y) != (x1, y1):
            if (x, y) != a and not self.is_free((x, y)):
                return False
            twice = 2 * err
            if twice >= dy:
                err += dy
                x += sx
            if twice <= dx:
                err += dx
                y += sy
        return self.is_free(b)

    def raster_base(self) -> np.ndarray:
        """Static map channels: traversable, hard-blocked, static cost, objectives."""
        out = np.zeros((4, self.height, self.width), dtype=np.float32)
        out[0] = (~self.hard_blocked).astype(np.float32)
        out[1] = self.hard_blocked.astype(np.float32)
        out[2] = self.static_cost
        for x, y, value in (*self._objective_marks(self.red_outpost, 1.0),
                            *self._objective_marks(self.blue_outpost, -1.0)):
            out[3, y, x] = value
        return out

    def _objective_marks(self, center: Cell, value: float) -> Iterable[tuple[int, int, float]]:
        cx, cy = center
        for x in range(max(0, cx - 1), min(self.width, cx + 2)):
            for y in range(max(0, cy - 1), min(self.height, cy + 2)):
                yield x, y, value
