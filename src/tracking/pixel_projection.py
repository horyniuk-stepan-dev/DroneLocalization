"""Project arbitrary frame pixels to map coordinates with the localizer state.

Same chain as Localizer.localize_optical_flow uses for the frame centre, but
for any point (e.g. the centre of a user-selected object):

    current frame px --S^-1--> keyframe px --*scale--> normalised
        --rot90--> rotated (H space) --H--> reference px --affine--> metric

``flow_affine`` is the 2x3 similarity KF->current in ORIGINAL pixels (identity
on the keyframe itself). ``None`` means the point is already in keyframe
coordinates.
"""

from __future__ import annotations

import numpy as np

from src.geometry.frame_transform import project_frame_points
from src.geometry.transformations import GeometryTransforms
from src.localization.rotation_geometry import _rotate_point_np90


def project_pixels_to_metric(
    state: dict,
    points_px: np.ndarray,
    frame_w: int,
    frame_h: int,
    flow_affine: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return Nx2 metric points, or None when the geometry is unusable."""
    if not state or state.get("H") is None or state.get("affine") is None:
        return None
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if flow_affine is not None:
        try:
            S3 = np.vstack([np.asarray(flow_affine, dtype=np.float64), [0.0, 0.0, 1.0]])
            S_inv = np.linalg.inv(S3)
        except (np.linalg.LinAlgError, ValueError):
            return None
        pts = pts @ S_inv[:2, :2].T + S_inv[:2, 2]

    scale = float(state.get("scale", 1.0) or 1.0)
    angle = int(state.get("global_angle", 0) or 0)
    w_n, h_n = frame_w * scale, frame_h * scale
    rot = np.array(
        [_rotate_point_np90(x * scale, y * scale, w_n, h_n, angle) for x, y in pts],
        dtype=np.float64,
    )
    try:
        pts_ref = GeometryTransforms.apply_homography(rot, state["H"])
        if pts_ref is None or len(pts_ref) == 0:
            return None
        ref_w, ref_h = state.get("reference_size", (0, 0))
        if ref_w and ref_h:
            metric = project_frame_points(state["affine"], pts_ref, ref_w, ref_h)
        else:
            metric = GeometryTransforms.apply_affine(pts_ref, state["affine"])
    except (ValueError, np.linalg.LinAlgError):
        return None
    if metric is None or not np.isfinite(metric).all():
        return None
    return np.asarray(metric, dtype=np.float64)
