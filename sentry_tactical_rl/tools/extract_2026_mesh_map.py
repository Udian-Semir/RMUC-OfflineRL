"""Export a top-down 2D annotation base from the Gazebo RMUC 2026 mesh.

The Gazebo SDF scales the DAE from millimetres to metres and shifts x by
-0.375 m.  This script applies that same transform and exports two images:

* ``rmuc2026_mesh_occupancy_candidate.png`` -- black = geometry above floor;
* ``rmuc2026_mesh_annotation_base.png`` -- same geometry with a 1 m grid.

They are annotation aids, not a certified navigation map.  Calibrate their
coordinates against three known referee landmarks before using them with the
28 x 15 m referee/radar frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh
import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Project the RMUC 2026 Gazebo mesh to a 2D annotation image")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out-dir", default="sentry_tactical_rl/assets")
    parser.add_argument("--resolution", type=float, default=0.02, help="metres per pixel")
    parser.add_argument("--horizontal-normal", type=float, default=0.90,
                        help="minimum |face normal z| kept in the top-down projection")
    args = parser.parse_args()
    if args.resolution <= 0:
        raise ValueError("resolution must be positive")

    scene = trimesh.load(args.mesh, force="scene")
    vertices = np.concatenate([np.asarray(mesh.vertices) for mesh in scene.geometry.values()], axis=0)
    # DAE values are mm; model.sdf supplies <scale>0.001</scale> and
    # <pose>-0.375 0 -0.2 ...</pose>.  z is shifted only for rendering.
    vertices *= 0.001
    vertices[:, 0] -= 0.375
    vertices[:, 2] -= 0.2
    min_xy = vertices[:, :2].min(axis=0)
    max_xy = vertices[:, :2].max(axis=0)
    size_xy = max_xy - min_xy
    width = int(np.ceil(size_xy[0] / args.resolution)) + 1
    height = int(np.ceil(size_xy[1] / args.resolution)) + 1
    # Project and fill faces rather than merely projecting vertices: the latter
    # leaves sparse dots on a triangulated DAE and is useless for annotation.
    # The DAE contains the physical field geometry rather than a separate floor
    # plane, so every projected face is rendered as geometry.  Traversability
    # (especially ramps) remains a later manual/sentry-specific annotation.
    # Vertical wall faces collapse to long diagonal lines under orthographic
    # projection.  Keep horizontal/near-horizontal surfaces to expose object
    # footprints and floor platforms cleanly; ramps are deliberately left for
    # manual sentry-traversability annotation.
    meshes = tuple(scene.geometry.values())
    if len(meshes) != 1:
        raise RuntimeError("the current exporter expects a single merged mesh")
    mesh = meshes[0]
    faces = np.asarray(mesh.faces)
    faces = faces[np.abs(np.asarray(mesh.face_normals)[:, 2]) >= args.horizontal_normal]
    xy = vertices[faces, :2]
    pixels = np.empty((len(faces), 3, 2), dtype=np.int32)
    pixels[:, :, 0] = np.clip(((xy[:, :, 0] - min_xy[0]) / args.resolution).astype(np.int32), 0, width - 1)
    pixels[:, :, 1] = np.clip(height - 1 - ((xy[:, :, 1] - min_xy[1]) / args.resolution).astype(np.int32), 0, height - 1)
    raw = np.full((height, width), 255, dtype=np.uint8)
    for start in range(0, len(pixels), 20_000):
        cv2.fillPoly(raw, pixels[start:start + 20_000], color=0, lineType=cv2.LINE_8)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "rmuc2026_mesh_occupancy_candidate.png"
    Image.fromarray(raw, mode="L").save(raw_path)

    overlay = Image.fromarray(np.repeat(raw[..., None], 3, axis=2), mode="RGB")
    draw = ImageDraw.Draw(overlay)
    pixels_per_m = 1.0 / args.resolution
    for metre in range(int(np.floor(size_xy[0])) + 1):
        x = int(round(metre * pixels_per_m))
        draw.line((x, 0, x, height), fill=(100, 150, 255), width=1)
        if metre % 2 == 0:
            draw.text((x + 2, 2), str(metre), fill=(0, 90, 220))
    for metre in range(int(np.floor(size_xy[1])) + 1):
        y = height - 1 - int(round(metre * pixels_per_m))
        draw.line((0, y, width, y), fill=(100, 150, 255), width=1)
        if metre % 2 == 0:
            draw.text((2, max(0, y - 12)), str(metre), fill=(0, 90, 220))
    overlay_path = out_dir / "rmuc2026_mesh_annotation_base.png"
    overlay.save(overlay_path)
    print(f"source bounds after SDF transform: x=[{min_xy[0]:.3f}, {max_xy[0]:.3f}], "
          f"y=[{min_xy[1]:.3f}, {max_xy[1]:.3f}]")
    print(f"wrote {raw_path} and {overlay_path}; resolution={args.resolution:.3f} m/px")


if __name__ == "__main__":
    main()
