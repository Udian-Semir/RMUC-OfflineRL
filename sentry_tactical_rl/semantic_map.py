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
import json
from pathlib import Path
from typing import Iterable

import cv2
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
    semantic_layers: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = (self.height, self.width)
        if self.hard_blocked.shape != shape or self.static_cost.shape != shape:
            raise ValueError(f"map layers must have shape {shape}")
        if np.any(self.static_cost < 0):
            raise ValueError("static_cost must be non-negative")
        for name, layer in self.semantic_layers.items():
            if layer.shape != shape:
                raise ValueError(f"semantic layer {name!r} must have shape {shape}")
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

    @classmethod
    def from_aligned_json(
        cls,
        path: str | Path,
        *,
        obstacle_path: str | Path | None = None,
        occupancy_threshold: float = 0.5,
    ) -> "SemanticMap":
        """Load the supplied RMUC semantic JSON onto the tactical grid.

        The source images are high-resolution annotation/occupancy layers;
        this adapter deliberately reduces them to the existing 1 m tactical
        grid.  Semantic masks stay separate from ``hard_blocked``.  The blue
        undulating-road layer is additionally treated as a sentry hard
        exclusion because that is the explicit annotation supplied for this
        policy.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        frame = payload.get("frame", {})
        field_x = float(frame.get("x_m", 28.0))
        field_y = float(frame.get("y_m", 15.0))
        width, height = int(round(field_x)), int(round(field_y))
        if not np.isclose(field_x, width) or not np.isclose(field_y, height):
            raise ValueError("aligned JSON frame must currently map to integer tactical metres")
        if not 0.0 <= occupancy_threshold <= 1.0:
            raise ValueError("occupancy_threshold must be in [0, 1]")

        if obstacle_path is None:
            obstacle_path = payload.get("obstacle_map", "blackwhite_map.png")
        obstacle = Path(obstacle_path)
        if not obstacle.is_absolute() and not obstacle.exists():
            candidate = path.parent / obstacle.name
            if candidate.exists():
                obstacle = candidate
        image = cv2.imread(str(obstacle), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(obstacle)
        rgb = image[:, :, :3][:, :, ::-1]
        blocked_source = np.all(rgb <= 5, axis=2)
        if image.ndim == 3 and image.shape[2] == 4:
            # Transparent padding is outside the usable field and must not
            # become a traversable route at the tactical grid boundary.
            blocked_source |= image[:, :, 3] == 0
        occupancy = cv2.resize(blocked_source.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
        hard = occupancy >= occupancy_threshold

        semantic_layers: dict[str, np.ndarray] = {}
        anchors: dict[str, Cell] = {}
        objective_centers: dict[str, list[tuple[float, float]]] = {"base": [], "outpost": []}
        for region in payload.get("regions", []):
            kind = str(region.get("kind", "unknown"))
            points = np.asarray(region.get("polygon_xy_m", []), dtype=np.float32)
            if len(points) < 3:
                continue
            px = np.rint(points[:, 0] / field_x * (width - 1)).astype(np.int32)
            py = np.rint((1.0 - points[:, 1] / field_y) * (height - 1)).astype(np.int32)
            mask = semantic_layers.setdefault(kind, np.zeros((height, width), dtype=bool))
            cv2.fillPoly(mask.view(np.uint8), [np.stack((px, py), axis=1)], 1)
            center = (float(points[:, 0].mean()), float(points[:, 1].mean()))
            if kind in objective_centers:
                objective_centers[kind].append(center)
            requested = (int(np.floor(center[0])), int(np.floor(center[1])))
            requested = (int(np.clip(requested[0], 0, width - 1)), int(np.clip(requested[1], 0, height - 1)))
            candidate = _nearest_free(hard, requested)
            if candidate is not None:
                anchors[str(region.get("id", f"{kind}_{len(anchors) + 1}"))] = candidate

        # Explicitly annotated as unusable by the sentry.  This is a policy
        # passability rule, not a claim that every robot has the same chassis.
        hard |= semantic_layers.get("undulating_road", np.zeros_like(hard))

        def objective(kind: str, side: str, fallback: Cell) -> Cell:
            centers = sorted(objective_centers.get(kind, []), key=lambda value: value[0])
            if not centers:
                return fallback
            point = centers[0] if side == "red" else centers[-1]
            return (
                int(np.clip(np.floor(point[0]), 0, width - 1)),
                int(np.clip(np.floor(point[1]), 0, height - 1)),
            )

        return cls(
            width=width,
            height=height,
            hard_blocked=hard,
            static_cost=np.zeros((height, width), dtype=np.float32),
            anchors=anchors,
            red_outpost=objective("outpost", "red", (5, 7)),
            blue_outpost=objective("outpost", "blue", (22, 7)),
            red_base=objective("base", "red", (1, 7)),
            blue_base=objective("base", "blue", (26, 7)),
            semantic_layers={name: layer.astype(np.float32) for name, layer in semantic_layers.items()},
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

    def nearest_free(self, cell: Cell, max_radius: int = 12) -> Cell | None:
        return _nearest_free(self.hard_blocked, cell, max_radius)

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
        """Static map channels plus independent semantic masks."""
        out = np.zeros((4 + len(self.semantic_layers), self.height, self.width), dtype=np.float32)
        out[0] = (~self.hard_blocked).astype(np.float32)
        out[1] = self.hard_blocked.astype(np.float32)
        out[2] = self.static_cost
        for x, y, value in (*self._objective_marks(self.red_outpost, 1.0),
                            *self._objective_marks(self.blue_outpost, -1.0)):
            out[3, y, x] = value
        for channel, name in enumerate(sorted(self.semantic_layers), start=4):
            out[channel] = self.semantic_layers[name]
        return out

    def _objective_marks(self, center: Cell, value: float) -> Iterable[tuple[int, int, float]]:
        cx, cy = center
        for x in range(max(0, cx - 1), min(self.width, cx + 2)):
            for y in range(max(0, cy - 1), min(self.height, cy + 2)):
                yield x, y, value


def _nearest_free(hard_blocked: np.ndarray, cell: Cell, max_radius: int = 12) -> Cell | None:
    """Find a nearby free cell without assuming a particular map topology."""
    x0 = int(np.clip(cell[0], 0, hard_blocked.shape[1] - 1))
    y0 = int(np.clip(cell[1], 0, hard_blocked.shape[0] - 1))
    if not hard_blocked[y0, x0]:
        return (x0, y0)
    for radius in range(1, max_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in (-radius, radius):
                x, y = x0 + dx, y0 + dy
                if 0 <= x < hard_blocked.shape[1] and 0 <= y < hard_blocked.shape[0] and not hard_blocked[y, x]:
                    return (x, y)
        for dy in range(-radius + 1, radius):
            for dx in (-radius, radius):
                x, y = x0 + dx, y0 + dy
                if 0 <= x < hard_blocked.shape[1] and 0 <= y < hard_blocked.shape[0] and not hard_blocked[y, x]:
                    return (x, y)
    return None
