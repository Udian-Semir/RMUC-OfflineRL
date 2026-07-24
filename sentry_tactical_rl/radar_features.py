"""Turn a radar cost-map snapshot into policy-ready candidate features.

No learned model appears here.  This is the deterministic radar-side bridge
between semantic goals/tracks and the RL policy: every candidate already carries
its conventional reachability, path cost and exposure estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .radar_costmap import CostMapSnapshot, RadarTrack, WorldPoint


@dataclass(frozen=True)
class SemanticAnchor:
    name: str
    position_m: WorldPoint
    skill_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class GoalCandidate:
    anchor: SemanticAnchor
    reachable: bool
    path_cost: float
    path_length_m: float
    mean_threat: float

    def feature_vector(self, cost_scale: float = 100.0) -> np.ndarray:
        """Stable candidate feature order used by future policy adapters."""
        return np.asarray((
            float(self.reachable),
            min(self.path_cost / cost_scale, 10.0) if self.reachable else 10.0,
            min(self.path_length_m / 30.0, 2.0) if self.reachable else 2.0,
            min(self.mean_threat, 5.0) if self.reachable else 5.0,
        ), dtype=np.float32)


class RadarFeatureBuilder:
    """Produces map channels and goal/target features from radar-side state."""

    def __init__(self, anchors: Iterable[SemanticAnchor]):
        self.anchors = tuple(anchors)
        if not self.anchors:
            raise ValueError("at least one semantic anchor is required")

    def goal_candidates(self, snapshot: CostMapSnapshot, sentry_position_m: WorldPoint) -> list[GoalCandidate]:
        candidates: list[GoalCandidate] = []
        for anchor in self.anchors:
            reachable, cost, path = snapshot.plan(sentry_position_m, anchor.position_m)
            if reachable:
                threat_samples = []
                for point in path:
                    x, y = snapshot.world_to_cell(point)
                    threat_samples.append(float(snapshot.threat_cost[y, x]))
                path_length = max(0, len(path) - 1) * snapshot.resolution_m
                mean_threat = float(np.mean(threat_samples)) if threat_samples else 0.0
            else:
                path_length, mean_threat = float("inf"), float("inf")
            candidates.append(GoalCandidate(anchor, reachable, float(cost), path_length, mean_threat))
        return candidates

    def build(self, snapshot: CostMapSnapshot, sentry_position_m: WorldPoint, enemy_tracks: Iterable[RadarTrack]) -> dict[str, np.ndarray]:
        """Return deterministic inputs that can be concatenated with RL state.

        ``map`` has channels [hard_blocked, static_cost, threat_cost,
        total_cost, sentry_position].  Enemy feature rows preserve caller roster
        order; an unavailable track should be omitted or have confidence zero.
        """
        candidates = self.goal_candidates(snapshot, sentry_position_m)
        raster = np.stack((
            snapshot.hard_blocked.astype(np.float32),
            snapshot.static_cost,
            snapshot.threat_cost,
            snapshot.total_cost,
            self._sentry_layer(snapshot, sentry_position_m),
        ))
        tracks = tuple(enemy_tracks)
        target_features = []
        for track in tracks:
            dx = (track.position_m[0] - sentry_position_m[0]) / max(snapshot.width * snapshot.resolution_m, 1e-6)
            dy = (track.position_m[1] - sentry_position_m[1]) / max(snapshot.height * snapshot.resolution_m, 1e-6)
            distance = float(np.hypot(track.position_m[0] - sentry_position_m[0], track.position_m[1] - sentry_position_m[1]))
            target_features.append((dx, dy, min(distance / 30.0, 2.0), float(np.clip(track.confidence, 0.0, 1.0)), float(track.alive)))
        return {
            "map": raster.astype(np.float32),
            "goal_features": np.stack([candidate.feature_vector() for candidate in candidates]),
            "goal_mask": np.asarray([candidate.reachable for candidate in candidates], dtype=bool),
            "target_features": np.asarray(target_features, dtype=np.float32).reshape(len(tracks), 5),
        }

    @staticmethod
    def _sentry_layer(snapshot: CostMapSnapshot, sentry_position_m: WorldPoint) -> np.ndarray:
        layer = np.zeros_like(snapshot.static_cost, dtype=np.float32)
        x, y = snapshot.world_to_cell(sentry_position_m)
        if snapshot.in_bounds((x, y)):
            layer[y, x] = 1.0
        return layer
