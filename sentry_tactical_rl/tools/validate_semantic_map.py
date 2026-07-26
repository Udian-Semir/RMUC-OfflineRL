"""Pixel-level validation for the colored semantic-map JSON.

The aligned JSON stores physical polygons, while the source annotation stores
exact RGB fills.  This tool projects both representations onto the obstacle
image grid and reports whether they describe the same regions.  It is a
geometry/annotation check only; overlap with black geometry is reported as
information because a tactical zone can surround or cover a physical object.

Example::

    python -m sentry_tactical_rl.tools.validate_semantic_map

The diagnostic image uses the following colors:

* green: source annotation and JSON agree
* red: source-only pixels (JSON missed them)
* blue: JSON-only pixels (polygon grew outside the source fill)
* yellow: pixels where source and JSON labels disagree
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


REGION_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "base": (255, 0, 0),
    "outpost": (0, 255, 255),
    "fortress": (255, 255, 0),
    "healing": (128, 0, 128),
    "central_highland": (51, 51, 51),
    "tunnel_gain": (255, 0, 255),
    "undulating_road": (0, 0, 255),
}
MIN_REGION_PIXELS = 500


def _read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _source_labels(annotation_rgb: np.ndarray, dst_shape: tuple[int, int],
                   regions: list[dict]) -> np.ndarray:
    """Return a region-instance label image projected with nearest sampling.

    The aligner numbers connected components within each kind (base_1,
    base_2, outpost_1, ...), so the validation labels must use that same
    category-local numbering rather than one global label per color.
    """
    dst_h, dst_w = dst_shape
    # Build the component masks at source resolution, then sample each mask on
    # the destination grid.  This exactly mirrors connectedComponentsWithStats
    # plus nearest-neighbour resize in align_semantic_map.py.
    labels = np.full((dst_h, dst_w), -1, dtype=np.int16)
    by_kind: dict[str, list[int]] = {}
    for index, region in enumerate(regions):
        by_kind.setdefault(str(region.get("kind", "")), []).append(index)
    for kind, rgb in REGION_COLORS_RGB.items():
        mask = np.all(annotation_rgb == np.asarray(rgb, dtype=np.uint8), axis=2).astype(np.uint8)
        count, connected, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        instance = 0
        region_indexes = by_kind.get(kind, [])
        for component in range(1, count):
            if int(stats[component, cv2.CC_STAT_AREA]) < MIN_REGION_PIXELS:
                continue
            if instance >= len(region_indexes):
                break
            component_mask = (connected == component).astype(np.uint8)
            resized = cv2.resize(component_mask, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            labels[resized] = region_indexes[instance]
            instance += 1
    return labels


def _json_labels(payload: dict, shape: tuple[int, int]) -> np.ndarray:
    dst_h, dst_w = shape
    frame = payload.get("frame", {})
    field_x = float(frame.get("x_m", 28.0))
    field_y = float(frame.get("y_m", 15.0))
    labels = np.full((dst_h, dst_w), -1, dtype=np.int16)
    for label, region in enumerate(payload.get("regions", [])):
        points = np.asarray(region.get("polygon_xy_m", []), dtype=np.float32)
        if len(points) < 3:
            continue
        # JSON uses lower-left physical coordinates; image rows increase down.
        px = np.rint(points[:, 0] / field_x * (dst_w - 1)).astype(np.int32)
        py = np.rint((1.0 - points[:, 1] / field_y) * (dst_h - 1)).astype(np.int32)
        polygon = np.stack((px, py), axis=1)
        cv2.fillPoly(labels, [polygon], int(label))
    return labels


def _diagnostic(source: np.ndarray, projected: np.ndarray) -> np.ndarray:
    """Build an RGB mismatch image; agreement is green."""
    out = np.zeros((*source.shape, 3), dtype=np.uint8)
    both = (source >= 0) & (projected >= 0)
    same = both & (source == projected)
    source_only = (source >= 0) & (projected < 0)
    json_only = (source < 0) & (projected >= 0)
    disagree = both & (source != projected)
    out[same] = (0, 200, 0)
    out[source_only] = (220, 0, 0)
    out[json_only] = (0, 90, 240)
    out[disagree] = (255, 210, 0)
    return out


def validate(annotation_path: str, obstacle_path: str, json_path: str,
             diagnostic_path: str, report_path: str) -> dict:
    annotation = _read_rgb(annotation_path)
    obstacle_bgr = cv2.imread(obstacle_path, cv2.IMREAD_UNCHANGED)
    if obstacle_bgr is None:
        raise FileNotFoundError(obstacle_path)
    obstacle_rgb = cv2.cvtColor(obstacle_bgr[:, :, :3], cv2.COLOR_BGR2RGB)
    shape = obstacle_rgb.shape[:2]
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    regions = payload.get("regions", [])
    source = _source_labels(annotation, shape, regions)
    projected = _json_labels(payload, shape)
    image = _diagnostic(source, projected)
    Path(diagnostic_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(diagnostic_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    rows = []
    for label, region in enumerate(regions):
        expected = source == label
        actual = projected == label
        intersection = int(np.count_nonzero(expected & actual))
        union = int(np.count_nonzero(expected | actual))
        source_pixels = int(np.count_nonzero(expected))
        json_pixels = int(np.count_nonzero(actual))
        # Black pixels are the mesh's geometry/obstacle candidate.  This is a
        # diagnostic statistic, not an automatic error: bases/outposts can be
        # drawn over their physical footprint and must later be split into
        # hard footprint plus traversable capture/gain area.
        obstacle_pixels = int(np.count_nonzero(actual & np.all(obstacle_rgb <= 5, axis=2)))
        rows.append({
            "id": region.get("id", f"region_{label}"),
            "kind": region.get("kind", "unknown"),
            "source_pixels": source_pixels,
            "json_pixels": json_pixels,
            "intersection_pixels": intersection,
            "iou": round(intersection / union, 6) if union else 1.0,
            "precision": round(intersection / json_pixels, 6) if json_pixels else 1.0,
            "recall": round(intersection / source_pixels, 6) if source_pixels else 1.0,
            "json_on_black_obstacle_pixels": obstacle_pixels,
            "json_on_black_obstacle_ratio": round(obstacle_pixels / json_pixels, 6) if json_pixels else 0.0,
        })

    source_any = source >= 0
    json_any = projected >= 0
    report = {
        "annotation": annotation_path,
        "obstacle": obstacle_path,
        "json": json_path,
        "image_shape_hw": list(shape),
        "region_count": len(regions),
        "source_unclassified_pixels": int(np.count_nonzero(~source_any)),
        "json_unclassified_pixels": int(np.count_nonzero(~json_any)),
        "source_only_pixels": int(np.count_nonzero(source_any & ~json_any)),
        "json_only_pixels": int(np.count_nonzero(~source_any & json_any)),
        "label_disagreement_pixels": int(np.count_nonzero(source_any & json_any & (source != projected))),
        "regions": rows,
        "diagnostic_legend": {
            "green": "source and JSON agree",
            "red": "source-only; JSON missed annotation pixels",
            "blue": "JSON-only; polygon grew beyond annotation",
            "yellow": "both classified but kind differs",
        },
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"shape={shape[1]}x{shape[0]} regions={len(regions)}")
    print(f"source_only={report['source_only_pixels']} json_only={report['json_only_pixels']} "
          f"label_disagreement={report['label_disagreement_pixels']}")
    for row in rows:
        print(f"{row['id']:<22} IoU={row['iou']:.4f} precision={row['precision']:.4f} "
              f"recall={row['recall']:.4f} black_overlap={row['json_on_black_obstacle_ratio']:.3f}")
    print(f"wrote {diagnostic_path} and {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate semantic-map JSON against its colored annotation")
    parser.add_argument("--annotation", default="sentry_tactical_rl/assets/demo_map.png")
    parser.add_argument("--obstacle", default="sentry_tactical_rl/assets/blackwhite_map.png")
    parser.add_argument("--json", dest="json_path", default="sentry_tactical_rl/assets/semantic_map_aligned.json")
    parser.add_argument("--diagnostic", default="sentry_tactical_rl/assets/semantic_map_debug.png")
    parser.add_argument("--report", default="sentry_tactical_rl/assets/semantic_map_validation.json")
    args = parser.parse_args()
    validate(args.annotation, args.obstacle, args.json_path, args.diagnostic, args.report)


if __name__ == "__main__":
    main()
