"""Keyframe selection primitives (pure Python, without torch/Qt).

Calculates inter-frame homography H and motion thresholds to decide if a frame
is a keyframe.
"""

from __future__ import annotations

import numpy as np

from src.geometry.transformations import GeometryTransforms

DEFAULT_MIN_TRANSLATION_PX = 15.0
DEFAULT_MIN_ROTATION_DEG = 1.5
DEFAULT_INTER_FRAME_MIN_MATCHES = 15
DEFAULT_INTER_FRAME_RANSAC_THRESH = 3.0


def normalize_homography(H: np.ndarray | None) -> np.ndarray | None:
    """Return a finite, full-rank homography in a deterministic projective gauge.

    Homographies that differ only by a non-zero scalar describe the same mapping.
    Keeping the arbitrary RANSAC scale in a long pose product makes determinant
    checks and overlap decisions depend on that scale, so every accepted matrix is
    canonicalised before it enters the pose chain.
    """
    if H is None:
        return None
    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return None

    norm = float(np.linalg.norm(matrix))
    if norm <= np.finfo(np.float64).eps:
        return None

    # Prefer the usual H[2, 2] gauge when it is numerically meaningful.  The
    # Frobenius fallback also handles valid homographies whose bottom-right entry
    # is zero (for example, a projective pole outside the image).
    if abs(matrix[2, 2]) > 1e-12 * norm:
        matrix = matrix / matrix[2, 2]
    else:
        matrix = matrix / norm
        pivot = np.unravel_index(np.argmax(np.abs(matrix)), matrix.shape)
        if matrix[pivot] < 0:
            matrix = -matrix

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values[-1] <= 1e-12 * singular_values[0]:
        return None
    return matrix


def is_significant_motion(
    H: np.ndarray,
    frame_w: int,
    frame_h: int,
    min_translation_px: float = DEFAULT_MIN_TRANSLATION_PX,
    min_rotation_deg: float = DEFAULT_MIN_ROTATION_DEG,
) -> bool:
    """Returns True if homography H (frame_b -> frame_a) indicates significant motion.

    Checks if frame center translation via H >= min_translation_px OR rotation angle >= min_rotation_deg.
    Degenerate H (|det| < 1e-6) returns True.
    """
    H = normalize_homography(H)
    if H is None:
        return True

    cx, cy = frame_w / 2.0, frame_h / 2.0
    p_src = np.array([cx, cy, 1.0], dtype=np.float64)
    p_dst = H.astype(np.float64) @ p_src
    if abs(p_dst[2]) <= 1e-12:
        return True
    p_dst /= p_dst[2]
    translation = np.linalg.norm(p_dst[:2] - np.array([cx, cy]))

    if translation >= min_translation_px:
        return True

    A = H[:2, :2].astype(np.float64)
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return True  # Degenerate matrix -> treat as motion
    angle_deg = abs(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    return bool(angle_deg >= min_rotation_deg)


def overlap_fraction(H: np.ndarray, frame_w: int, frame_h: int) -> float:
    """Returns overlap fraction between current frame and last keyframe in [0, 1].

    H is accumulated homography projecting current frame coordinates to last saved keyframe coordinates.
    """
    H = normalize_homography(H)
    if H is None:
        return 0.0

    w, h = float(frame_w), float(frame_h)
    if w <= 0 or h <= 0:
        return 0.0

    corners = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float64)
    homo = np.hstack([corners, np.ones((4, 1))])
    projected = homo @ H.T

    denominators = projected[:, 2]
    if (
        np.any(np.abs(denominators) <= 1e-12)
        or not np.all(np.isfinite(projected))
        or not (np.all(denominators > 0) or np.all(denominators < 0))
    ):
        return 0.0
    projected = projected[:, :2] / projected[:, 2:3]
    if not np.all(np.isfinite(projected)):
        return 0.0

    import cv2

    inter_area, _ = cv2.intersectConvexConvex(
        projected.astype(np.float32), corners.astype(np.float32)
    )
    frame_area = w * h
    if frame_area <= 0:
        return 0.0
    return float(np.clip(inter_area / frame_area, 0.0, 1.0))


def is_overlap_below(
    H: np.ndarray,
    frame_w: int,
    frame_h: int,
    max_overlap: float = 0.5,
) -> bool:
    """Returns True if overlap with last keyframe drops below max_overlap threshold."""
    return overlap_fraction(H, frame_w, frame_h) <= max_overlap


def compute_inter_frame_homography(
    matcher,
    fa: dict,
    fb: dict,
    *,
    min_matches: int = DEFAULT_INTER_FRAME_MIN_MATCHES,
    ransac_thresh: float = DEFAULT_INTER_FRAME_RANSAC_THRESH,
    homography_backend: str = "opencv",
    use_mad_ransac: bool = True,
    mad_k_factor: float = 2.5,
) -> np.ndarray | None:
    """Estimates H(fb -> fa) as 3x3 float64, returning None if match count is insufficient."""
    mkpts_a, mkpts_b = matcher.match(fa, fb)
    if len(mkpts_a) < min_matches:
        return None

    # matcher.match(fa, fb) returns corresponding points in the first and
    # second frame.  The pose chain consumes H(current=fb -> previous=fa).
    H, mask = GeometryTransforms.estimate_homography(
        mkpts_b,
        mkpts_a,
        ransac_threshold=ransac_thresh,
        backend=homography_backend,
        use_mad_ransac=use_mad_ransac,
        mad_k_factor=mad_k_factor,
    )

    if H is None or int(np.sum(mask)) < min_matches:
        return None

    return normalize_homography(H)
