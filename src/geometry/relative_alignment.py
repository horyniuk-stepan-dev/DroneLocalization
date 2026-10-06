"""Refine a LOCAL map using shared image points, with a fixed affine gauge.

The pose graph is an initializer. Its five-parameter relative transforms lose
shear, so minimizing their parameter residuals need not align image content.
This sparse affine adjustment minimizes the actual point discrepancies instead.
It still assumes approximately planar terrain; it cannot remove parallax.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree


@dataclass
class PointLink:
    first: int
    second: int
    first_points: np.ndarray
    second_points: np.ndarray
    weight: float = 1.0


def nearby_frame_pairs(affines, width, height, known_pairs, min_frame_gap, limit=8):
    """Bounded spatial proposals for overlaps missed by descriptor top-k retrieval.

    Nearby frames are proposals only: the caller must verify their image points.
    Allow a margin around estimated footprints because the initial map can drift.
    """
    if len(affines) < 2 or limit <= 0:
        return []
    ids = sorted(affines)
    center = np.array([width / 2, height / 2, 1.0])
    centers = np.array([affines[fid] @ center for fid in ids])
    radii = np.array(
        [
            0.5
            * (
                np.linalg.norm(affines[fid][:, 0]) * width
                + np.linalg.norm(affines[fid][:, 1]) * height
            )
            for fid in ids
        ]
    )
    tree = cKDTree(centers)
    known = {tuple(sorted(pair)) for pair in known_pairs}
    proposals = set()
    for index, fid in enumerate(ids):
        distances, neighbors = tree.query(centers[index], k=min(64, len(ids)))
        count = 0
        for distance, other in zip(distances, neighbors):
            key = tuple(sorted((fid, ids[other])))
            if (
                abs(fid - ids[other]) <= min_frame_gap
                or key in known
                or distance > 1.5 * (radii[index] + radii[other])
            ):
                continue
            proposals.add(key)
            count += 1
            if count >= limit:
                break
    return sorted(proposals)


def sample_point_pair(first, second, limit=96):
    """Keep a deterministic, spatially spread subset of verified matches."""
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if len(first) <= limit:
        return first.copy(), second.copy()
    selected = [int(np.argmin(first[:, 0]))]
    distances = np.full(len(first), np.inf)
    for _ in range(limit - 1):
        distances = np.minimum(distances, np.sum((first - first[selected[-1]]) ** 2, axis=1))
        distances[selected] = -1
        selected.append(int(np.argmax(distances)))
    return first[selected].copy(), second[selected].copy()


def refine_relative_affines(affines, links, width, height, gauge_id, check_running=lambda: None):
    """Return (affines, summary, per-frame RMS), without mutating the inputs.

    Huber IRLS limits surviving mismatches. Each link retains its graph weight,
    independent of the number of sampled points. Residuals use *fixed initial* pixel scales,
    so changing a frame's scale cannot redefine its error units. A weak shape
    prior stabilizes long chains; the first frame's entire affine is fixed.
    Invalid/disconnected solutions raise before the caller can save a map.
    """
    if width <= 0 or height <= 0 or gauge_id not in affines or len(affines) < 2:
        raise ValueError("Local point alignment needs dimensions, a gauge and connected frames")
    initial = {fid: np.asarray(a, dtype=np.float64).copy() for fid, a in affines.items()}
    for a in initial.values():
        if a.shape != (2, 3) or not np.isfinite(a).all() or abs(np.linalg.det(a[:, :2])) < 1e-12:
            raise ValueError("Invalid initial local affine")
    links = [link for link in links if link.first in initial and link.second in initial]
    neighbors = {fid: set() for fid in initial}
    for link in links:
        p, q = np.asarray(link.first_points), np.asarray(link.second_points)
        if not np.isfinite(link.weight) or link.weight <= 0:
            raise ValueError("Local point links need finite positive weights")
        if (
            p.ndim != 2
            or p.shape[1] != 2
            or p.shape != q.shape
            or len(p) < 3
            or not np.isfinite(p).all()
            or not np.isfinite(q).all()
            or np.linalg.matrix_rank(p - p.mean(axis=0)) < 2
            or np.linalg.matrix_rank(q - q.mean(axis=0)) < 2
        ):
            raise ValueError("Local point alignment needs non-collinear finite point pairs")
        neighbors[link.first].add(link.second)
        neighbors[link.second].add(link.first)
    reached, pending = {gauge_id}, [gauge_id]
    while pending:
        for fid in neighbors[pending.pop()] - reached:
            reached.add(fid)
            pending.append(fid)
    if reached != set(initial):
        raise ValueError("Local point matches do not connect every frame to the gauge")

    # Pixel basis centred on the image, with unit-sized coordinates. Solve for
    # corrections to the initial map, not large absolute map translations.
    basis = np.array([[width, 0, width / 2], [0, height, height / 2], [0, 0, 1.0]])
    inv_basis = np.linalg.inv(basis)
    ids = sorted(set(initial) - {gauge_id})
    columns = {fid: 3 * i for i, fid in enumerate(ids)}
    rows, cols, values, residuals, weights, slices = [], [], [], [], [], []
    start = 0
    weight_scale = float(np.median([link.weight for link in links]))
    for link in links:
        check_running()
        p, q = np.asarray(link.first_points), np.asarray(link.second_points)
        scale = np.sqrt(abs(np.linalg.det(initial[link.first][:, :2])))
        pa = np.column_stack([p, np.ones(len(p))])
        pb = np.column_stack([q, np.ones(len(q))])
        residuals.append((pa @ initial[link.first].T - pb @ initial[link.second].T) / scale)
        weights.extend([link.weight / (weight_scale * len(p))] * len(p))
        row_ids = np.arange(start, start + len(p))
        for fid, points, sign in ((link.first, pa, 1), (link.second, pb, -1)):
            if fid == gauge_id:
                continue
            normalized = points @ inv_basis.T * (sign / scale)
            rows.extend(np.repeat(row_ids, 3))
            cols.extend(np.tile(np.arange(columns[fid], columns[fid] + 3), len(p)))
            values.extend(normalized.ravel())
        slices.append(slice(start, start + len(p)))
        start += len(p)
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(start, 3 * len(ids)))
    residual = np.concatenate(residuals)
    base_weight = np.asarray(weights)
    # A 100-pixel change at the image edge costs about one squared pixel per
    # frame. Translation is practically unregularized; connectivity fixes it.
    prior = np.concatenate(
        [np.array([1e-4, 1e-4, 1e-10]) / abs(np.linalg.det(initial[fid][:, :2])) for fid in ids]
    )
    correction = np.zeros((3 * len(ids), 2))
    for iteration in range(12):
        check_running()
        norms = np.linalg.norm(residual + matrix @ correction, axis=1)
        robust_weight = np.minimum(1.0, 3.0 / np.maximum(norms, 1e-12))
        weighted = matrix.multiply((base_weight * robust_weight)[:, None])
        normal = (matrix.T @ weighted + sparse.diags(prior)).tocsc()
        updated = spsolve(normal, -(weighted.T @ residual))
        if not np.isfinite(updated).all():
            raise ValueError("Local point alignment produced a non-finite solution")
        change = np.max(np.abs(matrix @ (updated - correction)))
        correction = updated
        if change < 1e-4:
            break
    result = {fid: a.copy() for fid, a in initial.items()}
    for fid in ids:
        offset = columns[fid]
        result[fid] += correction[offset : offset + 3].T @ inv_basis
        relative = result[fid][:, :2] @ np.linalg.inv(initial[fid][:, :2])
        singular = np.linalg.svd(relative, compute_uv=False)
        if np.linalg.det(relative) <= 0 or singular.min() < 0.25 or singular.max() > 4:
            raise ValueError("Local point alignment would fold or excessively distort the map")
    before = np.linalg.norm(residual, axis=1)
    after = np.linalg.norm(residual + matrix @ correction, axis=1)

    def huber_cost(errors):
        return float(np.sum(base_weight * np.where(errors <= 3, errors**2 / 2, 3 * errors - 4.5)))

    if huber_cost(after) > huber_cost(before) + 1e-7:
        raise ValueError("Local point alignment did not improve point consistency")
    frame_errors = {fid: [] for fid in initial}
    for link, section in zip(links, slices):
        # Per-frame confidence uses that frame's own pixel scale.
        ratio = np.sqrt(
            abs(
                np.linalg.det(initial[link.first][:, :2])
                / np.linalg.det(initial[link.second][:, :2])
            )
        )
        frame_errors[link.first].extend(after[section] ** 2)
        frame_errors[link.second].extend((after[section] * ratio) ** 2)
    summary = {
        "method": "affine_point_irls_v1",
        "units": "initial_reference_pixels",
        "frames": len(initial),
        "links": len(links),
        "points": len(after),
        "iterations": iteration + 1,
        "before_median_px": float(np.median(before)),
        "after_median_px": float(np.median(after)),
        "before_p95_px": float(np.percentile(before, 95)),
        "after_p95_px": float(np.percentile(after, 95)),
    }
    return (
        result,
        summary,
        {fid: float(np.sqrt(np.mean(errors))) for fid, errors in frame_errors.items()},
    )
