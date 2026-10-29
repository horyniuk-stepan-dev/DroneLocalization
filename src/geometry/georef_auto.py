"""Automatic control points for georeferencing a local (no-GPS) map.

Probe frames of the local map are localized against an already geographic
layer (the reference map). Every successful fix is a candidate control point:
the frame centre in local units <-> the fix in projected metres. Single fixes
can be confidently wrong (repetitive texture), so candidates are accepted only
when they agree with one another on a common similarity — the shape of the
local map is known, so wrong fixes stand out as not fitting it.

Pure numpy, no Qt / models: the localization itself is done by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.geometry.georef_fit import _as_points, apply, fit_similarity


def select_probe_frames(centers, frame_ids, budget: int) -> np.ndarray:
    """Frame ids spread over the map: farthest-point sampling in local units,
    seeded at the two ends of the flight (where gauge leverage is largest)."""
    centers = _as_points(centers)
    frame_ids = np.asarray(frame_ids)
    n = len(centers)
    if budget <= 0 or n == 0:
        return np.array([], dtype=frame_ids.dtype)
    if budget >= n:
        return np.sort(frame_ids)
    order = np.argsort(frame_ids, kind="stable")
    chosen = [int(order[0])]
    if budget > 1:
        chosen.append(int(order[-1]))
    dist = np.min(
        [np.linalg.norm(centers - centers[c], axis=1) for c in chosen], axis=0
    )
    while len(chosen) < budget:
        nxt = int(np.argmax(dist))
        if dist[nxt] <= 0:
            break
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(centers - centers[nxt], axis=1))
    return np.sort(frame_ids[chosen])


@dataclass
class Consensus:
    matrix: np.ndarray | None  # 2x3 similarity on the inliers (None: no consensus)
    inliers: np.ndarray  # bool per candidate
    errors: np.ndarray  # distance to the consensus similarity, projected units
    tolerance: float  # projected units used for the inlier test
    warnings: list[str] = field(default_factory=list)

    @property
    def n_inliers(self) -> int:
        return int(self.inliers.sum())


def consensus_similarity(
    local_pts,
    metric_pts,
    *,
    rotation_sign: int = 1,
    tolerance: float,
    drift_frac: float = 0.05,
    weights=None,
    min_inliers: int = 3,
) -> Consensus:
    """RANSAC over candidate pairs for the similarity most candidates agree on.

    The inlier test allows ``tolerance + drift_frac * d``, where ``d`` is the
    distance (projected units) from the candidate to the nearest of the two
    points that generated the hypothesis: the local map drifts in scale, so a
    rigid fit legitimately degrades away from where it was fixed. All pairs are
    tried (candidates are few), the hypothesis with the largest weighted
    support wins, then it is refined on its inliers.
    """
    src, dst = _as_points(local_pts), _as_points(metric_pts)
    n = len(src)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    empty = Consensus(None, np.zeros(n, bool), np.full(n, np.inf), float(tolerance))
    if n < 2:
        empty.warnings.append("менше двох успішних збігів — консенсусу немає")
        return empty

    def inliers_of(M, base):
        err = np.linalg.norm(apply(M, src) - dst, axis=1)
        d = np.min([np.linalg.norm(dst - dst[b], axis=1) for b in base], axis=0)
        return err, err <= tolerance + drift_frac * d

    best = None
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(src[i] - src[j]) <= 1e-9:
                continue
            try:
                M = fit_similarity(src[[i, j]], dst[[i, j]], rotation_sign)
            except ValueError:
                continue
            err, ok = inliers_of(M, (i, j))
            score = (float(w[ok].sum()), -float(np.sum(err[ok])))
            if best is None or score > best[0]:
                best = (score, (i, j), ok)
    if best is None:
        empty.warnings.append("збіги вироджені (точки збігаються)")
        return empty

    ok = best[2]
    M = fit_similarity(src[ok], dst[ok], rotation_sign)
    for _ in range(3):  # refine: support of the inlier fit, base = all inliers
        base = tuple(np.flatnonzero(ok))
        err, new_ok = inliers_of(M, base)
        if new_ok.sum() < 2 or np.array_equal(new_ok, ok):
            break
        ok = new_ok
        M = fit_similarity(src[ok], dst[ok], rotation_sign)
    err = np.linalg.norm(apply(M, src) - dst, axis=1)

    out = Consensus(M, ok, err, float(tolerance))
    if ok.sum() < min_inliers:
        out.warnings.append(
            f"узгоджених збігів лише {int(ok.sum())} — прив’язка ненадійна, "
            "перевірте точки вручну або додайте свої"
        )
    if ok.sum() >= 2:
        sv = np.linalg.svd(src[ok] - src[ok].mean(axis=0), compute_uv=False)
        if sv[0] <= 0 or sv[1] / sv[0] < 0.1:
            out.warnings.append("узгоджені збіги лежать майже на одній лінії")
    return out
