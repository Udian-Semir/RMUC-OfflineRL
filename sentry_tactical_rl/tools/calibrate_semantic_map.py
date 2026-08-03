"""Fit and report a source-image-pixel to radar/map-coordinate transform.

The supplied semantic PNG uses ordinary top-left image pixels.  Fill the
corresponding real radar/referee positions in ``radar_landmark_template.json``
and run this tool before treating the 2D tactical map as physically aligned.
It only writes a transform report; applying it to navigation/semantic data is
a separate reviewed integration step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def fit_affine(source_xy: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
    """Return a homogeneous 3x3 affine transform mapping source to world."""
    if source_xy.shape != world_xy.shape or source_xy.ndim != 2 or source_xy.shape[1] != 2:
        raise ValueError("source_xy and world_xy must both have shape [N, 2]")
    if source_xy.shape[0] < 3:
        raise ValueError("an affine transform needs at least three landmarks")
    design = np.column_stack((source_xy, np.ones(source_xy.shape[0])))
    coefficients, _, rank, _ = np.linalg.lstsq(design, world_xy, rcond=None)
    if rank < 3:
        raise ValueError("landmarks are collinear or duplicated; use non-collinear points")
    return np.asarray(
        [[coefficients[0, 0], coefficients[1, 0], coefficients[2, 0]],
         [coefficients[0, 1], coefficients[1, 1], coefficients[2, 1]],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def fit_homography(source_xy: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
    """Return a homogeneous projective transform for four or more landmarks."""
    if source_xy.shape != world_xy.shape or source_xy.ndim != 2 or source_xy.shape[1] != 2:
        raise ValueError("source_xy and world_xy must both have shape [N, 2]")
    if source_xy.shape[0] < 4:
        raise ValueError("a homography needs at least four landmarks")
    rows: list[list[float]] = []
    for (u, v), (x, y) in zip(source_xy, world_xy):
        rows.append([u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v, -x])
        rows.append([0.0, 0.0, 0.0, u, v, 1.0, -y * u, -y * v, -y])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    matrix = vt[-1].reshape(3, 3)
    if abs(matrix[2, 2]) < 1e-12:
        raise ValueError("degenerate homography; choose better spread landmarks")
    return matrix / matrix[2, 2]


def transform_points(matrix: np.ndarray, source_xy: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 3x3 matrix to [N, 2] points."""
    matrix = np.asarray(matrix, dtype=np.float64)
    source_xy = np.asarray(source_xy, dtype=np.float64)
    if matrix.shape != (3, 3) or source_xy.ndim != 2 or source_xy.shape[1] != 2:
        raise ValueError("expected matrix [3, 3] and source points [N, 2]")
    homogeneous = np.column_stack((source_xy, np.ones(source_xy.shape[0])))
    projected = homogeneous @ matrix.T
    if np.any(np.isclose(projected[:, 2], 0.0)):
        raise ValueError("transform projects a landmark to infinity")
    return projected[:, :2] / projected[:, 2:3]


def fit_calibration(landmarks: list[dict[str, Any]], *, model: str) -> dict[str, Any]:
    """Fit a calibration report from fully specified landmark records."""
    missing = [str(item.get("id", "<unnamed>")) for item in landmarks if item.get("world_xy_m") is None]
    if missing:
        raise ValueError("world_xy_m is missing for: " + ", ".join(missing))
    source = np.asarray([item["pixel_xy"] for item in landmarks], dtype=np.float64)
    world = np.asarray([item["world_xy_m"] for item in landmarks], dtype=np.float64)
    if source.shape != world.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("each landmark needs numeric pixel_xy and world_xy_m pairs")
    matrix = fit_affine(source, world) if model == "affine" else fit_homography(source, world)
    predicted = transform_points(matrix, source)
    residuals = np.linalg.norm(predicted - world, axis=1)
    records = []
    for item, estimate, residual in zip(landmarks, predicted, residuals):
        records.append({
            "id": item.get("id"),
            "pixel_xy": [float(value) for value in item["pixel_xy"]],
            "world_xy_m": [float(value) for value in item["world_xy_m"]],
            "predicted_world_xy_m": [float(value) for value in estimate],
            "residual_m": float(residual),
        })
    return {
        "model": model,
        "matrix_source_pixel_to_world": matrix.tolist(),
        "landmark_count": len(landmarks),
        "rmse_m": float(np.sqrt(np.mean(np.square(residuals)))),
        "max_residual_m": float(np.max(residuals)),
        "landmarks": records,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="fit pixel-to-radar semantic-map calibration")
    parser.add_argument(
        "--landmarks",
        default=str(root / "sentry_tactical_rl/assets/radar_landmark_template.json"),
        help="JSON containing pixel_xy and measured world_xy_m landmark pairs",
    )
    parser.add_argument("--model", choices=("affine", "homography"), default="affine")
    parser.add_argument(
        "--out",
        default=str(root / "sentry_tactical_rl/assets/semantic_map_calibration.json"),
        help="output transform and residual report",
    )
    args = parser.parse_args()
    landmark_path = Path(args.landmarks)
    with landmark_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    landmarks = payload.get("landmarks", [])
    report = fit_calibration(landmarks, model=args.model)
    report["source_landmarks"] = str(landmark_path)
    report["source_image"] = payload.get("source_image")
    report["target_frame"] = payload.get("target_frame", "map")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    print(
        f"calibration written to {out}: model={report['model']} landmarks={report['landmark_count']} "
        f"rmse={report['rmse_m']:.4f}m max={report['max_residual_m']:.4f}m"
    )


if __name__ == "__main__":
    main()
