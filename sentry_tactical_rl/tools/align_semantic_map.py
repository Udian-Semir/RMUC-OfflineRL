"""Project colored semantic regions onto the static black/white obstacle map.

The annotation image and obstacle image have different pixel resolutions but
share the 28 x 15 m RMUC field frame.  Regions are therefore transformed
through physical coordinates, not copied as pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


FIELD_X = 28.0
FIELD_Y = 15.0
MIN_REGION_PIXELS = 500

# Exact fill colors in demo_map.png.  The pale (100,150,255) grid is deliberately
# absent: it is a drawing aid, not a semantic zone.
REGIONS = (
    ("base", (255, 0, 0), {}),
    ("outpost", (0, 255, 255), {}),
    ("fortress", (255, 255, 0), {}),
    ("healing", (128, 0, 128), {}),
    ("central_highland", (51, 51, 51), {}),
    ("tunnel_gain", (255, 0, 255), {}),
    # The user explicitly marked this terrain as present but unusable by the
    # sentry; make it an action/path-planning hard exclusion from day one.
    ("undulating_road", (0, 0, 255), {"sentry_passable": False}),
)


def _mask_for_color(rgb: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    return np.all(rgb == np.asarray(color, np.uint8), axis=2).astype(np.uint8)


def _physical_polygon(contour: np.ndarray, src_w: int, src_h: int):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, max(1.0, perimeter * 0.005), True).reshape(-1, 2)
    out = []
    for px, py in approx:
        x = float(px) / max(src_w - 1, 1) * FIELD_X
        y = (1.0 - float(py) / max(src_h - 1, 1)) * FIELD_Y
        out.append([round(x, 3), round(y, 3)])
    return out


def build(annotation_path: str, obstacle_path: str, out_image: str, out_json: str,
          opacity: float = 0.78):
    annotation = cv2.imread(annotation_path, cv2.IMREAD_COLOR)
    obstacle = cv2.imread(obstacle_path, cv2.IMREAD_UNCHANGED)
    if annotation is None:
        raise FileNotFoundError(annotation_path)
    if obstacle is None:
        raise FileNotFoundError(obstacle_path)
    if obstacle.ndim == 2:
        obstacle = cv2.cvtColor(obstacle, cv2.COLOR_GRAY2BGRA)
    elif obstacle.shape[2] == 3:
        obstacle = cv2.cvtColor(obstacle, cv2.COLOR_BGR2BGRA)

    src_rgb = cv2.cvtColor(annotation, cv2.COLOR_BGR2RGB)
    src_h, src_w = src_rgb.shape[:2]
    dst_h, dst_w = obstacle.shape[:2]
    out = obstacle[:, :, :3].astype(np.float32)
    metadata = []

    for semantic, rgb, props in REGIONS:
        src_mask = _mask_for_color(src_rgb, rgb)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(src_mask, 8)
        region_num = 0
        for component in range(1, n):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < MIN_REGION_PIXELS:
                continue
            component_mask = (labels == component).astype(np.uint8)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            region_num += 1
            physical = _physical_polygon(contour, src_w, src_h)
            dst_mask = cv2.resize(component_mask, (dst_w, dst_h),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            # cv2 is BGR while the annotation specification is RGB.
            bgr = np.asarray(rgb[::-1], np.float32)
            out[dst_mask] = out[dst_mask] * (1.0 - opacity) + bgr * opacity
            x, y, w, h = stats[component, :4]
            metadata.append({
                "id": f"{semantic}_{region_num}",
                "kind": semantic,
                "polygon_xy_m": physical,
                "source_bbox_px": [int(x), int(y), int(w), int(h)],
                **props,
            })

    output = np.dstack((np.clip(out, 0, 255).astype(np.uint8), obstacle[:, :, 3]))
    Path(out_image).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_image, output)
    payload = {
        "frame": {"x_m": FIELD_X, "y_m": FIELD_Y, "origin": "lower_left"},
        "obstacle_map": str(obstacle_path),
        "annotation_map": str(annotation_path),
        "regions": metadata,
        "note": "Zone effects, capture timing, and rewards must be filled from the rule model; this file only records geometry and the explicit sentry exclusion.",
    }
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"wrote {out_image} and {out_json}; regions={len(metadata)}")
    for region in metadata:
        print(f"  {region['id']}: {region['kind']} points={len(region['polygon_xy_m'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", default="sentry_tactical_rl/assets/demo_map.png")
    ap.add_argument("--obstacle", default="sentry_tactical_rl/assets/blackwhite_map.png")
    ap.add_argument("--out-image", default="sentry_tactical_rl/assets/semantic_map_aligned.png")
    ap.add_argument("--out-json", default="sentry_tactical_rl/assets/semantic_map_aligned.json")
    ap.add_argument("--opacity", type=float, default=0.78)
    args = ap.parse_args()
    if not 0.0 <= args.opacity <= 1.0:
        raise SystemExit("--opacity must be in [0, 1]")
    build(args.annotation, args.obstacle, args.out_image, args.out_json, args.opacity)


if __name__ == "__main__":
    main()
