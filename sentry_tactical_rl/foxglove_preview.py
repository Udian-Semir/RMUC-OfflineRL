"""Pure geometry/image preparation for the local Foxglove semantic preview.

The preview deliberately preserves the source PNG's pixel geometry.  The
current semantic JSON uses a provisional 28 x 15 m frame while its occupancy
image is 1400 x 774 pixels, so a regular ROS OccupancyGrid cannot represent both
axes with one physical resolution without a small display-only y transform.
This module exposes that transform explicitly.  It is never used by planning,
reward, or navigation; those continue to consume the JSON world coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


Color = tuple[float, float, float, float]
Point = tuple[float, float]

KIND_COLORS: dict[str, Color] = {
    "base": (1.0, 0.15, 0.15, 0.95),
    "outpost": (0.0, 0.95, 0.95, 0.95),
    "fortress": (1.0, 0.9, 0.0, 0.95),
    "healing": (0.75, 0.15, 0.85, 0.95),
    "central_highland": (0.2, 0.65, 0.25, 0.95),
    "tunnel_gain": (1.0, 0.0, 0.9, 0.95),
    "undulating_road": (0.15, 0.4, 1.0, 0.95),
}
DEFAULT_COLOR: Color = (0.95, 0.95, 0.95, 0.95)


@dataclass(frozen=True)
class PreviewTransform:
    """Map provisional JSON coordinates to the pixel-faithful preview frame."""

    resolution_m: float
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float

    def point(self, xy_m: Point) -> Point:
        return (
            xy_m[0] * self.scale_x + self.offset_x,
            xy_m[1] * self.scale_y + self.offset_y,
        )

    def source_point(self, preview_xy_m: Point) -> Point:
        """Convert a cropped Foxglove preview point back to JSON coordinates."""
        return (
            (preview_xy_m[0] - self.offset_x) / self.scale_x,
            (preview_xy_m[1] - self.offset_y) / self.scale_y,
        )


@dataclass(frozen=True)
class SemanticRegionPreview:
    region_id: str
    kind: str
    points_m: tuple[Point, ...]
    color: Color

    @property
    def label_position_m(self) -> Point:
        points = np.asarray(self.points_m, dtype=np.float32)
        return (float(points[:, 0].mean()), float(points[:, 1].mean()))


@dataclass(frozen=True)
class SemanticPreview:
    """OccupancyGrid payload plus polygon specs independent of ROS imports."""

    width: int
    height: int
    resolution_m: float
    occupancy: np.ndarray  # [H, W], values {-1, 0, 100}, row 0 is lower y
    transform: PreviewTransform
    regions: tuple[SemanticRegionPreview, ...]
    json_path: Path
    obstacle_path: Path
    crop_box_px: tuple[int, int, int, int]  # left, top, right, bottom in the source PNG


def load_semantic_preview(json_path: str | Path, obstacle_path: str | Path | None = None) -> SemanticPreview:
    """Load a project JSON/PNG pair and prepare a pixel-faithful preview."""
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    obstacle = _resolve_obstacle_path(json_path, payload, obstacle_path)
    rgba = np.asarray(Image.open(obstacle).convert("RGBA"), dtype=np.uint8)
    source_height, source_width = rgba.shape[:2]
    frame = payload.get("frame", {})
    field_x = float(frame.get("x_m", 28.0))
    field_y = float(frame.get("y_m", 15.0))
    if source_width <= 0 or source_height <= 0 or field_x <= 0.0 or field_y <= 0.0:
        raise ValueError("semantic preview requires a non-empty image and positive frame extents")

    # The source drawing has a transparent external frame.  Foxglove renders
    # unknown OccupancyGrid cells grey, which leaves a false grey border around
    # the physical field.  Crop only that frame for preview; the original PNG
    # remains the RL/navigation input and semantic points are translated below.
    left, top, right, bottom = _opaque_crop_bounds(rgba)
    rgba = rgba[top:bottom, left:right]
    height, width = rgba.shape[:2]

    # OccupancyGrid has square cells.  Use x as its display scale, retain all
    # image rows, and scale marker y coordinates by the corresponding ratio so
    # their outlines land on exactly the pixels that produced the JSON.
    resolution = field_x / source_width
    transform = PreviewTransform(
        resolution_m=resolution,
        scale_x=(source_width * resolution) / field_x,
        scale_y=(source_height * resolution) / field_y,
        offset_x=-left * resolution,
        # PNG rows increase downwards while JSON y increases up.  Removing the
        # bottom source rows shifts marker y values by this amount.
        offset_y=-(source_height - bottom) * resolution,
    )
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    occupied_top_down = np.all(rgb <= 5, axis=2)
    grid_top_down = np.where(alpha == 0, -1, np.where(occupied_top_down, 100, 0)).astype(np.int8)
    occupancy = np.flipud(grid_top_down).copy()

    regions: list[SemanticRegionPreview] = []
    for index, raw in enumerate(payload.get("regions", [])):
        points = raw.get("polygon_xy_m", [])
        if len(points) < 3:
            continue
        converted = tuple(transform.point((float(x), float(y))) for x, y in points)
        kind = str(raw.get("kind", "unknown"))
        regions.append(SemanticRegionPreview(
            region_id=str(raw.get("id", f"region_{index}")),
            kind=kind,
            points_m=converted,
            color=KIND_COLORS.get(kind, DEFAULT_COLOR),
        ))
    return SemanticPreview(
        width=width,
        height=height,
        resolution_m=resolution,
        occupancy=occupancy,
        transform=transform,
        regions=tuple(regions),
        json_path=json_path,
        obstacle_path=obstacle,
        crop_box_px=(left, top, right, bottom),
    )


def _resolve_obstacle_path(json_path: Path, payload: dict[str, Any], override: str | Path | None) -> Path:
    candidate = Path(override) if override is not None else Path(str(payload.get("obstacle_map", "blackwhite_map.png")))
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    sibling = json_path.parent / candidate.name
    if sibling.exists():
        return sibling
    raise FileNotFoundError(candidate)


def _opaque_crop_bounds(rgba: np.ndarray) -> tuple[int, int, int, int]:
    """Return the bounding box of visible pixels, trimming only external padding."""
    # Preserve the slightly anti-aliased field boundary.  It belongs to the
    # map, unlike the completely transparent external frame that Foxglove
    # would otherwise render as a broad unknown/grey border.
    visible = rgba[:, :, 3] > 0
    rows, cols = np.where(visible)
    if not rows.size:
        raise ValueError("semantic preview image is fully transparent")
    return int(cols.min()), int(rows.min()), int(cols.max() + 1), int(rows.max() + 1)
