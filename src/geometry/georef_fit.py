"""Fit a local (no-GPS) map onto projected geographic coordinates from control points.

Pure numpy, no Qt / h5py. A LOCAL map built from inter-frame links has an
arbitrary translation, rotation and scale; a handful of ground control points
(GCPs: a map point in local X/Y plus its true lat/lon) fixes that gauge.

All coordinates here are planar ``[x, y]``: local map units on the source side,
projected metres (the layer's CoordinateConverter metric) on the target side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SIMILARITY = "similarity"
AFFINE = "affine"
MIN_POINTS = {SIMILARITY: 2, AFFINE: 4}


@dataclass
class GeorefFit:
    matrix: np.ndarray  # 2x3: local [x, y] -> projected metric [x, y]
    kind: str
    residuals: np.ndarray  # per point, projected units
    loo_residuals: np.ndarray | None = None  # leave-one-out check error, projected units
    warnings: list[str] = field(default_factory=list)

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2))) if len(self.residuals) else 0.0

    @property
    def scale(self) -> float:
        """Projected units per local unit (geometric mean of the axes)."""
        return float(np.sqrt(abs(np.linalg.det(self.matrix[:, :2]))))

    @property
    def rotation_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.matrix[1, 0], self.matrix[0, 0])))

    @property
    def reflected(self) -> bool:
        return bool(np.linalg.det(self.matrix[:, :2]) < 0)


def _as_points(a) -> np.ndarray:
    pts = np.asarray(a, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or not np.isfinite(pts).all():
        raise ValueError("points must be a finite (N, 2) array")
    return pts


def fit_similarity(src, dst, rotation_sign: int = 1) -> np.ndarray:
    """Least-squares similarity (Umeyama) src -> dst.

    ``rotation_sign`` = +1 forces a proper rotation, -1 forces a mirror. The
    sign is a property of the map, not of the points, so it is decided by the
    caller (two points cannot tell a mirror from a rotation).
    """
    src, dst = _as_points(src), _as_points(dst)
    if len(src) != len(dst) or len(src) < 2:
        raise ValueError("need >= 2 matching point pairs")
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s_c, d_c = src - mu_s, dst - mu_d
    var_s = float(np.mean(np.sum(s_c**2, axis=1)))
    if var_s <= 1e-18:
        raise ValueError("control points coincide")
    cov = d_c.T @ s_c / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = 1.0 if np.linalg.det(U) * np.linalg.det(Vt) > 0 else -1.0
    D = np.diag([1.0, d * (1.0 if rotation_sign >= 0 else -1.0)])
    R = U @ D @ Vt
    scale = float(np.trace(np.diag(S) @ D)) / var_s
    if scale <= 1e-12:
        raise ValueError("control points contradict the map orientation (mirrored?)")
    t = mu_d - scale * R @ mu_s
    return np.hstack([scale * R, t[:, None]])


def fit_affine(src, dst) -> np.ndarray:
    src, dst = _as_points(src), _as_points(dst)
    if len(src) != len(dst) or len(src) < 3:
        raise ValueError("need >= 3 matching point pairs")
    A = np.hstack([src, np.ones((len(src), 1))])
    sol, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return sol.T


def apply(matrix, pts) -> np.ndarray:
    pts = _as_points(pts)
    return pts @ np.asarray(matrix)[:, :2].T + np.asarray(matrix)[:, 2]


def geometry_warnings(src) -> list[str]:
    """Point layouts that make the fit unreliable even with a small RMSE."""
    src = _as_points(src)
    out = []
    if len(src) < 3:
        out.append("2 точки: розв’язок точний, але не перевірений — додайте третю")
        return out
    sv = np.linalg.svd(src - src.mean(axis=0), compute_uv=False)
    if sv[0] <= 0 or sv[1] / sv[0] < 0.1:
        out.append("точки майже на одній лінії — поворот/масштаб погано обумовлені")
    return out


def fit_georef(src, dst, kind: str = SIMILARITY, rotation_sign: int = 1) -> GeorefFit:
    src, dst = _as_points(src), _as_points(dst)
    if kind not in MIN_POINTS:
        raise ValueError(f"unknown fit kind {kind!r}")
    if len(src) < MIN_POINTS[kind]:
        raise ValueError(f"{kind}: need >= {MIN_POINTS[kind]} points, have {len(src)}")

    def solve(s, d):
        return fit_similarity(s, d, rotation_sign) if kind == SIMILARITY else fit_affine(s, d)

    M = solve(src, dst)
    residuals = np.linalg.norm(apply(M, src) - dst, axis=1)
    loo = None
    if len(src) > MIN_POINTS[kind]:
        loo = np.empty(len(src))
        for i in range(len(src)):
            keep = np.arange(len(src)) != i
            try:
                Mi = solve(src[keep], dst[keep])
                loo[i] = float(np.linalg.norm(apply(Mi, src[i : i + 1])[0] - dst[i]))
            except ValueError:
                loo[i] = np.nan
    warnings = geometry_warnings(src)
    if kind == SIMILARITY and len(src) >= 3:
        # Swapped lat/lon mirrors the points; the forced handedness then fits badly.
        try:
            alt = np.linalg.norm(apply(fit_similarity(src, dst, -rotation_sign), src) - dst, axis=1)
            if np.sqrt(np.mean(alt**2)) < 0.5 * np.sqrt(np.mean(residuals**2)):
                warnings.append("точки відповідають дзеркальній карті — чи не переплутані lat/lon?")
        except ValueError:
            pass
    return GeorefFit(M, kind, residuals, loo, warnings)


def required_rotation_sign(local_affines, target_det_sign: float = -1.0) -> int:
    """Handedness the fit must have so the georeferenced frames keep the
    geographic convention (pixel y down -> north up, det < 0)."""
    dets = [np.linalg.det(np.asarray(a)[:2, :2]) for a in local_affines]
    dets = [d for d in dets if np.isfinite(d) and d != 0]
    if not dets:
        return 1
    local_sign = 1.0 if np.median(dets) > 0 else -1.0
    return 1 if local_sign * target_det_sign > 0 else -1


def compose_affine(M, A) -> np.ndarray:
    """2x3 ∘ 2x3: first frame pixels -> local (A), then local -> metric (M)."""
    M3 = np.vstack([np.asarray(M, dtype=np.float64), [0.0, 0.0, 1.0]])
    A3 = np.vstack([np.asarray(A, dtype=np.float64)[:2], [0.0, 0.0, 1.0]])
    return (M3 @ A3)[:2]


def compose_homography(M, H) -> np.ndarray:
    M3 = np.vstack([np.asarray(M, dtype=np.float64), [0.0, 0.0, 1.0]])
    return M3 @ np.asarray(H, dtype=np.float64)
