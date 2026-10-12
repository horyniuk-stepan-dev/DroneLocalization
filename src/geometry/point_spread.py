"""Spatial spread of correspondences — ill-conditioning of geometric estimation.

Motivation (OrthoTrack §3.4, `docs/RESEARCH_ADDENDUM_2026-07.md` item 1): inliers
may be formally sufficient in count, but if all of them are clustered in one corner,
the homography/affine estimation is ill-conditioned — the model extrapolates onto the rest
of the frame, sending the frame center to a bad coordinate. Inlier count cannot detect this,
nor can RANSAC — a locally consistent cluster is a valid consensus.

Pure functions (numpy only) — testable in any environment and called
from both sides: live (`ResultBuilder.compute_confidence`) and offline
(`PropagationPipeline._match_and_build_edge`).

IMPORTANT semantics: low spread is NOT automatically an error. At the boundary of DB
coverage, query frame legitimately overlaps with reference only by a corner, and clustering
there is correct. Therefore, the metric enters the pipeline continuously (multiplier to
confidence / edge weight), while a hard gate remains only at the extreme.

IMPORTANT about None: ``None`` means "signal unavailable" (too few points, corrupted
frame), whereas ``0.0`` means "spread is truly zero" (all points on a single line —
worst possible case). Do not confuse them: the former should not be penalized,
the latter should be penalized maximally.
"""

from __future__ import annotations

import numpy as np

# Spread of a uniform distribution over side L: σ = L/√12 ≈ 0.2887·L.
# So a healthy nadir frame with uniform coverage gives spread ≈ 0.29.
UNIFORM_SPREAD = float(1.0 / np.sqrt(12.0))


def inlier_spread(
    points: np.ndarray, frame_w: float, frame_h: float, *, principal=False
) -> float | None:
    """Axis spread by default; principal spread for offline accuracy validation.

    Normalizing by ``min(W, H)`` makes the metric invariant to both resolution and
    aspect ratio: for uniform coverage sigma_x = 0.2887*W, sigma_y = 0.2887*H,
    so min(sigma_x, sigma_y) = 0.2887*min(W, H) -> spread ≈ 0.289 for any aspect.

    With ``principal=True``, uses covariance eigenvalues so diagonal and horizontal
    lines are equally degenerate. The default preserves the live confidence
    metric (minimum of the two image-axis standard deviations).

    Args:
        points: (N, 2) pixel coordinates in query frame.
        frame_w: frame width in the same pixels as ``points``.
        frame_h: frame height.

    Returns:
        Spread in [0, ~0.5], or ``None`` if metric cannot be computed
        (< 2 valid points, zero/non-finite frame dimensions). Zero is a
        valid value ("completely degenerate cloud"), not signal absence.
    """
    if points is None:
        return None
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return None

    denom = min(float(frame_w), float(frame_h))
    if not np.isfinite(denom) or denom <= 0.0:
        return None

    xy = pts[:, :2]
    if not np.all(np.isfinite(xy)):
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if xy.shape[0] < 2:
            return None

    if not principal:
        return float(np.min(np.std(xy, axis=0)) / denom)
    centered = xy - xy.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(centered))
    # Roundoff must not turn an exactly collinear cloud into weak 2D support.
    smallest = max(0.0, float(eigenvalues[0]))
    if smallest <= np.finfo(float).eps * max(float(eigenvalues[-1]), 1.0) * 8:
        return 0.0
    return float(np.sqrt(smallest) / denom)


def spread_confidence_factor(
    spread: float | None, spread_ref: float = 0.15, floor: float = 0.35
) -> float:
    """Multiplier for live localization confidence: clip(spread/ref, floor, 1.0).

    ``spread_ref`` = 0.15 — approximately half of uniform coverage (0.289):
    above this there is no penalty at all. ``floor`` prevents confidence from zeroing out —
    a clustered fix remains a measurement with larger R in Kalman, rather than
    a discarded frame (which is the difference from OrthoTrack hard threshold).

    ``None`` (signal unavailable) -> 1.0, no penalty. ``0.0`` (degenerate
    cloud) -> ``floor``, maximum penalty.
    """
    if spread is None or not np.isfinite(spread):
        return 1.0
    ref = max(float(spread_ref), 1e-6)
    return float(np.clip(max(0.0, float(spread)) / ref, float(floor), 1.0))


def spread_weight_factor(spread: float | None, spread_ref: float = 0.15, k: float = 10.0) -> float:
    """Multiplier to graph edge weight: 1 / (1 + k*max(0, ref - spread)).

    Same form as used for affine fit quality (``temporal_weight_use_fit_quality``),
    so weights stay on the same scale.
    Spread deficit is bounded above by ``ref`` (0.15), so k should be around 10
    for penalty to be noticeable: spread 0.05 -> x0.50, spread 0 -> x0.40.

    ``None`` (signal unavailable) -> 1.0, no penalty.
    """
    if spread is None or not np.isfinite(spread):
        return 1.0
    deficit = max(0.0, float(spread_ref) - max(0.0, float(spread)))
    return float(1.0 / (1.0 + float(k) * deficit))
