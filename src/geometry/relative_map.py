"""Image-motion maps in arbitrary units; no geographic or metric scale implied."""

import numpy as np

from src.geometry.frame_transform import as_homography, project_points, valid_footprint


def normalize_relative_map(affines, width, height, extent=100.0):
    """Fit all connected frame footprints into a square, preserving aspect ratio.

    Returns transformed affines and the single similarity applied to the map.
    Never stretch x/y independently: that would corrupt subsequent geometry.
    """
    if not affines or width <= 0 or height <= 0 or extent <= 0:
        raise ValueError("A relative map needs frames and positive dimensions")
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    if any(not valid_footprint(a, width, height) for a in affines.values()):
        raise ValueError("Invalid relative map footprint")
    footprints = np.concatenate([project_points(a, corners) for a in affines.values()])
    if not np.isfinite(footprints).all():
        raise ValueError("Non-finite relative map geometry")
    lo, hi = footprints.min(axis=0), footprints.max(axis=0)
    span = float(np.max(hi - lo))
    if span <= 1e-9:
        raise ValueError("Degenerate relative map")
    scale = extent / span
    offset = (extent - (hi - lo) * scale) / 2 - lo * scale
    transform = np.array([[scale, 0, offset[0]], [0, scale, offset[1]], [0, 0, 1]])
    result = {
        fid: (transform @ as_homography(a))[: 2 if np.shape(a) == (2, 3) else 3]
        for fid, a in affines.items()
    }
    return result, transform
