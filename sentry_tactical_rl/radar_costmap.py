"""Radar-side semantic and dynamic cost-map construction.

This module belongs on the radar/decision side, not on the sentry.  It turns a
surveyed ROS static map plus radar tracks into a conservative traversability
mask, a dynamic threat field and path-cost queries.  The sentry still performs
its own final local ESDF safety check before executing any received goal.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import yaml


GridCell = tuple[int, int]  # (x, y), with y increasing in ROS map coordinates
WorldPoint = tuple[float, float]


@dataclass(frozen=True)
class RadarTrack:
    """A radar track used for threat projection, not an omniscient simulator state."""

    position_m: WorldPoint
    confidence: float = 1.0
    threat_weight: float = 1.0
    alive: bool = True


@dataclass(frozen=True)
class SemanticCostZone:
    """A radial soft-cost layer; positive is avoidance, negative is preference."""

    center_m: WorldPoint
    radius_m: float
    cost: float


@dataclass
class CostMapSnapshot:
    resolution_m: float
    origin_m: WorldPoint
    hard_blocked: np.ndarray  # [H, W], already inflated for the sentry footprint
    static_cost: np.ndarray
    threat_cost: np.ndarray

    @property
    def total_cost(self) -> np.ndarray:
        return self.static_cost + self.threat_cost

    @property
    def width(self) -> int:
        return int(self.hard_blocked.shape[1])

    @property
    def height(self) -> int:
        return int(self.hard_blocked.shape[0])

    def in_bounds(self, cell: GridCell) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def is_free(self, cell: GridCell) -> bool:
        return self.in_bounds(cell) and not bool(self.hard_blocked[cell[1], cell[0]])

    def world_to_cell(self, point_m: WorldPoint) -> GridCell:
        return (int(np.floor((point_m[0] - self.origin_m[0]) / self.resolution_m)),
                int(np.floor((point_m[1] - self.origin_m[1]) / self.resolution_m)))

    def cell_to_world(self, cell: GridCell) -> WorldPoint:
        return (self.origin_m[0] + (cell[0] + 0.5) * self.resolution_m,
                self.origin_m[1] + (cell[1] + 0.5) * self.resolution_m)

    def nearest_free(self, cell: GridCell, max_radius_cells: int = 12) -> GridCell | None:
        if self.is_free(cell):
            return cell
        for radius in range(1, max_radius_cells + 1):
            for dx in range(-radius, radius + 1):
                for dy in (-radius, radius):
                    candidate = (cell[0] + dx, cell[1] + dy)
                    if self.is_free(candidate):
                        return candidate
            for dy in range(-radius + 1, radius):
                for dx in (-radius, radius):
                    candidate = (cell[0] + dx, cell[1] + dy)
                    if self.is_free(candidate):
                        return candidate
        return None

    def plan(self, start_m: WorldPoint, goal_m: WorldPoint) -> tuple[bool, float, list[WorldPoint]]:
        """A* query over hard constraints and the composed radar-side cost."""
        start = self.nearest_free(self.world_to_cell(start_m))
        goal = self.nearest_free(self.world_to_cell(goal_m))
        if start is None or goal is None:
            return False, float("inf"), []
        queue: list[tuple[float, float, GridCell]] = [(0.0, 0.0, start)]
        came_from: dict[GridCell, GridCell | None] = {start: None}
        g_score: dict[GridCell, float] = {start: 0.0}
        while queue:
            _, current_g, current = heapq.heappop(queue)
            if current_g != g_score.get(current):
                continue
            if current == goal:
                cells: list[GridCell] = []
                node: GridCell | None = current
                while node is not None:
                    cells.append(node)
                    node = came_from[node]
                cells.reverse()
                return True, current_g, [self.cell_to_world(cell) for cell in cells]
            for nxt, step_cost in self._neighbours(current):
                nx, ny = nxt
                candidate = current_g + step_cost + float(self.total_cost[ny, nx])
                if candidate >= g_score.get(nxt, float("inf")):
                    continue
                g_score[nxt] = candidate
                came_from[nxt] = current
                heuristic = float(np.hypot(goal[0] - nx, goal[1] - ny))
                heapq.heappush(queue, (candidate + heuristic, candidate, nxt))
        return False, float("inf"), []

    def _neighbours(self, cell: GridCell) -> Iterable[tuple[GridCell, float]]:
        x, y = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nxt = (x + dx, y + dy)
            if not self.is_free(nxt):
                continue
            # Do not allow diagonal corner cutting through two obstacles.
            if dx and dy and (not self.is_free((x + dx, y)) or not self.is_free((x, y + dy))):
                continue
            yield nxt, 1.41421356 if dx and dy else 1.0


class RadarCostMap:
    """Immutable static map plus methods to compose a fresh radar snapshot."""

    def __init__(self, resolution_m: float, origin_m: WorldPoint, hard_blocked: np.ndarray, static_cost: np.ndarray) -> None:
        self.resolution_m = float(resolution_m)
        self.origin_m = tuple(float(value) for value in origin_m)
        self.hard_blocked = hard_blocked.astype(bool, copy=True)
        self.static_cost = static_cost.astype(np.float32, copy=True)
        if self.hard_blocked.shape != self.static_cost.shape:
            raise ValueError("hard_blocked and static_cost shapes differ")

    @classmethod
    def from_ros_map_yaml(
        cls,
        yaml_path: str | Path,
        *,
        target_resolution_m: float = 0.10,
        sentry_radius_m: float = 0.35,
        obstacle_proximity_cost: float = 0.55,
        proximity_m: float = 0.75,
    ) -> "RadarCostMap":
        """Load a standard ROS map YAML/PNG conservatively.

        A coarser tactical grid is formed by marking an output cell blocked if
        any source pixel in it is occupied *or unknown*.  The blocked layer is
        then inflated by the sentry radius.  This preserves the radar-side rule
        that a learned policy may never select an impossible corridor.
        """
        yaml_path = Path(yaml_path)
        with yaml_path.open("r", encoding="utf-8") as handle:
            meta = yaml.safe_load(handle)
        source_resolution = float(meta["resolution"])
        ratio = target_resolution_m / source_resolution
        rounded_ratio = int(round(ratio))
        if not np.isclose(ratio, rounded_ratio):
            raise ValueError("target_resolution_m must be an integer multiple of ROS map resolution")
        image_path = yaml_path.parent / str(meta["image"])
        pixels = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
        # Image rows are top->bottom, while ROS coordinates use y increasing
        # from the YAML origin; flip before pooling.
        pixels = np.flipud(pixels)
        if pixels.shape[0] % rounded_ratio or pixels.shape[1] % rounded_ratio:
            raise ValueError("map dimensions must divide exactly at target resolution")
        occupied_threshold = float(meta.get("occupied_thresh", 0.65))
        free_threshold = float(meta.get("free_thresh", 0.25))
        negate = int(meta.get("negate", 0))
        occupancy = pixels.astype(np.float32) / 255.0
        occupancy = occupancy if negate else 1.0 - occupancy
        free = occupancy <= free_threshold
        # Midrange pixels are unknown/anti-aliased; treating them as blocked is
        # conservative and consistent with an on-robot safety boundary.
        source_blocked = ~free
        h, w = source_blocked.shape
        pooled = source_blocked.reshape(h // rounded_ratio, rounded_ratio, w // rounded_ratio, rounded_ratio)
        hard = pooled.any(axis=(1, 3))
        inflation_cells = int(np.ceil(sentry_radius_m / target_resolution_m))
        hard = _dilate(hard, inflation_cells)
        proximity_cells = int(np.ceil(proximity_m / target_resolution_m))
        near_obstacle = _dilate(hard, proximity_cells)
        static_cost = np.where(near_obstacle & ~hard, obstacle_proximity_cost, 0.0).astype(np.float32)
        origin = tuple(meta.get("origin", (0.0, 0.0, 0.0))[:2])
        return cls(target_resolution_m, origin, hard, static_cost)

    def snapshot(
        self,
        enemy_tracks: Iterable[RadarTrack],
        *,
        semantic_zones: Iterable[SemanticCostZone] = (),
        threat_decay_m: float = 2.8,
    ) -> CostMapSnapshot:
        """Compose dynamic threat and semantic soft-cost fields for one tick."""
        yy, xx = np.mgrid[0:self.hard_blocked.shape[0], 0:self.hard_blocked.shape[1]]
        world_x = self.origin_m[0] + (xx + 0.5) * self.resolution_m
        world_y = self.origin_m[1] + (yy + 0.5) * self.resolution_m
        threat = np.zeros_like(self.static_cost)
        for track in enemy_tracks:
            if not track.alive or track.confidence <= 0.0:
                continue
            distance = np.hypot(world_x - track.position_m[0], world_y - track.position_m[1])
            threat += float(np.clip(track.confidence, 0.0, 1.0) * track.threat_weight) * np.exp(-distance / threat_decay_m)
        semantic = np.zeros_like(self.static_cost)
        for zone in semantic_zones:
            if zone.radius_m <= 0.0:
                continue
            distance = np.hypot(world_x - zone.center_m[0], world_y - zone.center_m[1])
            semantic += np.where(distance <= zone.radius_m, zone.cost * (1.0 - distance / zone.radius_m), 0.0)
        # A negative semantic preference must not make a path edge have a
        # negative total cost; A* assumes non-negative edge weights.
        static = np.maximum(self.static_cost + semantic, 0.0).astype(np.float32)
        threat[self.hard_blocked] = 0.0
        return CostMapSnapshot(self.resolution_m, self.origin_m, self.hard_blocked, static, threat.astype(np.float32))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Euclidean binary dilation using only NumPy; maps are small (112x60)."""
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            src_x0, src_x1 = max(0, -dx), min(w, w - dx)
            src_y0, src_y1 = max(0, -dy), min(h, h - dy)
            dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
            dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
            out[dst_y0:dst_y1, dst_x0:dst_x1] |= mask[src_y0:src_y1, src_x0:src_x1]
    return out
