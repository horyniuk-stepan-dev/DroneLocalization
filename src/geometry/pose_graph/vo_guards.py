"""Temporal Visual Odometry (VO) guards.

Sanity checks for temporal graph edges to detect degenerate transformations (wild rotation/scale/shifts)
and consistent aliasing across un-anchored gaps. Pure Python functions without Qt/PyTorch dependencies.
"""

from __future__ import annotations

import numpy as np

from src.geometry.affine_utils import decompose_affine_5dof
from src.geometry.pose_graph.model_5dof import (
    GraphEdge,
    _predict_forward,
    _predict_inverse,
)


def temporal_edge_sane(
    similarity_2x3: np.ndarray,
    gap: int,
    frame_w: int,
    frame_h: int,
    max_rotation_deg: float = 30.0,
    max_scale_ratio: float = 1.4,
    max_shift_frac: float = 1.2,
) -> tuple[bool, str]:
    """Sanity bounds for a SINGLE temporal edge. Returns (ok, reason)."""
    M = np.asarray(similarity_2x3, dtype=np.float64)
    _, _, sx, sy, angle = decompose_affine_5dof(M)

    rot_deg = abs(float(np.degrees(angle)))
    if rot_deg > max_rotation_deg:
        return False, f"rotation {rot_deg:.1f}° > {max_rotation_deg:.0f}°"

    max_log = float(np.log(max(max_scale_ratio, 1.0 + 1e-9)))
    if abs(np.log(max(sx, 1e-9))) > max_log or abs(np.log(max(sy, 1e-9))) > max_log:
        return (
            False,
            f"scale ({sx:.3f},{sy:.3f}) out of bounds [1/{max_scale_ratio},{max_scale_ratio}]",
        )

    cx, cy = frame_w / 2.0, frame_h / 2.0
    dcx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2] - cx
    dcy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2] - cy
    shift = float(np.hypot(dcx, dcy))
    diag = float(np.hypot(frame_w, frame_h))
    limit = max_shift_frac * diag * max(int(gap), 1)
    if shift > limit:
        return False, f"|Δcenter| {shift:.0f}px > {limit:.0f}px (gap={gap})"

    return True, ""


def check_anchor_gaps(
    edges: list[GraphEdge],
    anchor_states: dict[int, np.ndarray],
    sign: float,
    max_dev_m: float = 150.0,
) -> dict[tuple[int, int], dict]:
    """Checks the temporal chain in gaps between ADJACENT anchors.

    For anchor pair (a, b): starts from state of anchor a, composes
    temporal edges to b and compares predicted center with anchor b's center.

    Statuses: "ok" — deviation <= max_dev_m; "inconsistent" — chain
    is complete but deviates; "broken" — missing edge inside gap.
    """
    ids = sorted(anchor_states)
    report: dict[tuple[int, int], dict] = {}
    if len(ids) < 2:
        return report

    # For each node — forward temporal edge with minimal jump
    fwd: dict[int, GraphEdge] = {}
    for e in edges:
        if e.edge_type != "temporal":
            continue
        lo, hi = (e.from_id, e.to_id) if e.from_id < e.to_id else (e.to_id, e.from_id)
        cur = fwd.get(lo)
        if cur is None or hi < max(cur.from_id, cur.to_id):
            fwd[lo] = e

    for a, b in zip(ids, ids[1:]):
        state = np.array(anchor_states[a], dtype=np.float64).copy()
        cur, n_edges, broken = a, 0, False
        while cur < b:
            e = fwd.get(cur)
            nxt = None if e is None else max(e.from_id, e.to_id)
            if e is None or nxt is None or nxt > b:
                broken = True
                break
            if e.from_id == cur:
                state = _predict_forward(state, e, sign)
            else:
                state = _predict_inverse(state, e, sign)
            cur = nxt
            n_edges += 1

        if broken:
            report[(a, b)] = {"status": "broken", "dev_m": None, "n_edges": n_edges}
            continue

        dev = float(np.linalg.norm(state[:2] - np.asarray(anchor_states[b])[:2]))
        report[(a, b)] = {
            "status": "inconsistent" if dev > max_dev_m else "ok",
            "dev_m": dev,
            "n_edges": n_edges,
        }
    return report


def downweight_gap_edges(
    edges: list[GraphEdge],
    gap_pairs: list[tuple[int, int]],
    factor: float = 0.05,
) -> int:
    """Applies weight factor to temporal edges within specified gaps."""
    if not gap_pairs:
        return 0
    n = 0
    for e in edges:
        if e.edge_type != "temporal":
            continue
        lo, hi = min(e.from_id, e.to_id), max(e.from_id, e.to_id)
        for a, b in gap_pairs:
            if lo >= a and hi <= b:
                e.weight *= factor
                n += 1
                break
    return n


def select_gap_fallback_frames(
    results_centers: dict[int, tuple[float, float]],
    anchor_states: dict[int, np.ndarray],
    flagged_gaps: list[tuple[int, int]],
    max_dev_m: float = 150.0,
) -> set[int]:
    """Selects candidate frames in flagged gaps whose centers deviate beyond threshold."""
    out: set[int] = set()
    for a, b in flagged_gaps:
        if a not in anchor_states or b not in anchor_states or b - a < 2:
            continue
        ca = np.asarray(anchor_states[a][:2], dtype=np.float64)
        cb = np.asarray(anchor_states[b][:2], dtype=np.float64)
        for f in range(a + 1, b):
            c = results_centers.get(f)
            if c is None:
                continue
            t = (f - a) / (b - a)
            ref = ca + t * (cb - ca)
            if float(np.hypot(c[0] - ref[0], c[1] - ref[1])) > max_dev_m:
                out.add(f)
    return out
