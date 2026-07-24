"""Validate ROS-map loading and radar-side path-cost queries."""
from __future__ import annotations

import argparse

import numpy as np

from .radar_costmap import RadarCostMap, RadarTrack
from .radar_features import RadarFeatureBuilder, SemanticAnchor


def _largest_component(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return the largest 4-connected free component without scipy."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        queue, component = [(int(x), int(y))], []
        seen[y, x] = True
        while queue:
            cx, cy = queue.pop()
            component.append((cx, cy))
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((nx, ny))
        if len(component) > len(best):
            best = component
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Check radar-side ROS map and cost-map planning")
    parser.add_argument("--map-yaml", required=True)
    parser.add_argument("--sentry-radius", type=float, default=0.35)
    args = parser.parse_args()
    costmap = RadarCostMap.from_ros_map_yaml(args.map_yaml, sentry_radius_m=args.sentry_radius)
    snapshot = costmap.snapshot([RadarTrack((14.0, 7.5), confidence=0.8)])
    component = _largest_component(~snapshot.hard_blocked)
    if len(component) < 2:
        raise RuntimeError("no usable free-space component after obstacle inflation")
    start_cell = min(component, key=lambda cell: cell[0])
    goal_cell = max(component, key=lambda cell: cell[0])
    ok, cost, path = snapshot.plan(snapshot.cell_to_world(start_cell), snapshot.cell_to_world(goal_cell))
    if not ok:
        raise RuntimeError("A* failed inside the largest free-space component")
    features = RadarFeatureBuilder((
        SemanticAnchor("component_start", snapshot.cell_to_world(start_cell)),
        SemanticAnchor("component_goal", snapshot.cell_to_world(goal_cell)),
    )).build(snapshot, snapshot.cell_to_world(start_cell), [RadarTrack((14.0, 7.5), confidence=0.8)])
    if features["map"].shape[0] != 5 or not features["goal_mask"].any():
        raise RuntimeError("failed to construct policy-ready radar features")
    print("costmap ok: shape={} resolution={:.2f}m blocked={:.1%} largest_component={} path_points={} cost={:.2f}".format(
        snapshot.hard_blocked.shape, snapshot.resolution_m, snapshot.hard_blocked.mean(), len(component), len(path), cost,
    ))


if __name__ == "__main__":
    main()
