import numpy as np

from simulation2d.mapping.costmap import RadarCostMap, RadarTrack
from simulation2d.mapping.features import RadarFeatureBuilder, SemanticAnchor


def test_dynamic_costmap_builds_policy_features() -> None:
    blocked = np.zeros((15, 28), dtype=bool)
    blocked[4:11, 13] = True
    static_cost = np.zeros_like(blocked, dtype=np.float32)
    costmap = RadarCostMap(1.0, (0.0, 0.0), blocked, static_cost)
    tracks = [RadarTrack((18.5, 7.5), confidence=0.8)]
    snapshot = costmap.snapshot(tracks)
    builder = RadarFeatureBuilder((SemanticAnchor("safe", (4.5, 7.5)),))

    features = builder.build(snapshot, (2.5, 7.5), tracks)

    assert features["map"].shape == (5, 15, 28)
    assert features["goal_mask"].tolist() == [True]
    assert features["target_features"].shape == (1, 5)
    assert snapshot.threat_cost.max() > 0.0

