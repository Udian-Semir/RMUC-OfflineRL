"""Export the exact sentry-centre obstacle raster consumed by tactical A*."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ..world.semantic_map import SemanticMap


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a pre-inflated A* obstacle PNG")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--obstacle-map", type=Path, required=True)
    parser.add_argument("--clearance-m", type=float, default=0.35)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tactical_map = SemanticMap.from_aligned_json(
        args.json,
        obstacle_path=args.obstacle_map,
        sentry_clearance_m=args.clearance_m,
        use_preinflated_astar=False,
    )
    # hard_blocked is stored with y increasing upward; PNG rows increase down.
    image = np.where(np.flipud(tactical_map.hard_blocked), 0, 255).astype(np.uint8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), image):
        raise RuntimeError(f"failed to write {args.out}")
    print(
        f"wrote={args.out} size={tactical_map.width}x{tactical_map.height} "
        f"resolution_m={tactical_map.resolution_m:.3f} clearance_m={args.clearance_m:.3f} "
        f"blocked_fraction={tactical_map.hard_blocked.mean():.4f}"
    )


if __name__ == "__main__":
    main()
