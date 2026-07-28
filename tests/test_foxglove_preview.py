"""No-ROS tests for preparing the project-local Foxglove preview payload."""
from __future__ import annotations

from pathlib import Path
import unittest

from sentry_tactical_rl.foxglove_preview import load_semantic_preview


class FoxglovePreviewTest(unittest.TestCase):
    def test_project_assets_become_an_occupancy_grid_and_closed_region_specs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        preview = load_semantic_preview(root / "sentry_tactical_rl/assets/semantic_map_aligned.json")

        self.assertEqual((preview.width, preview.height), (1352, 726))
        self.assertEqual(preview.occupancy.shape, (726, 1352))
        self.assertEqual(preview.occupancy.dtype.name, "int8")
        self.assertEqual(preview.crop_box_px, (22, 23, 1374, 749))
        self.assertEqual(len(preview.regions), 15)
        self.assertAlmostEqual(preview.resolution_m, 0.02)
        self.assertGreater(preview.transform.scale_y, 1.0)
        self.assertAlmostEqual(preview.transform.offset_x, -0.44)
        self.assertAlmostEqual(preview.transform.offset_y, -0.50)
        self.assertEqual(
            tuple(round(value, 3) for value in preview.transform.source_point(preview.transform.point((12.345, 6.789)))),
            (12.345, 6.789),
        )
        self.assertTrue(all(len(region.points_m) >= 3 for region in preview.regions))


if __name__ == "__main__":
    unittest.main()
