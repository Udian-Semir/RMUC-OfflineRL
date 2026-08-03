"""Tests for the deterministic pixel-to-radar calibration fit."""
from __future__ import annotations

import unittest

import numpy as np

from sentry_tactical_rl.tools.calibrate_semantic_map import fit_calibration, transform_points


class SemanticMapCalibrationTest(unittest.TestCase):
    def test_affine_fit_recovers_known_transform_and_residuals(self) -> None:
        matrix = np.asarray([
            [0.02, -0.001, 1.5],
            [0.002, -0.02, 15.2],
            [0.0, 0.0, 1.0],
        ])
        pixels = np.asarray([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [350.0, 220.0]])
        world = transform_points(matrix, pixels)
        landmarks = [
            {"id": f"landmark_{index}", "pixel_xy": pixel.tolist(), "world_xy_m": point.tolist()}
            for index, (pixel, point) in enumerate(zip(pixels, world))
        ]

        report = fit_calibration(landmarks, model="affine")

        self.assertLess(report["rmse_m"], 1e-10)
        self.assertLess(report["max_residual_m"], 1e-10)
        self.assertTrue(np.allclose(np.asarray(report["matrix_source_pixel_to_world"]), matrix, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
