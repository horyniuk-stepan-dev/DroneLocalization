"""Conservative anchor-only motion model for straight survey legs.

This is an explicit kinematic assumption, not a substitute for visual geometry on
arbitrary flights.  Three consecutive long anchor intervals must agree on their
per-slot velocity before any interior slot is generated from the anchors.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from src.geometry.pose_graph.model_5dof import _affine_to_state, _state_to_affine


def linear_anchor_intervals(
    anchor_affines: Mapping[int, np.ndarray],
    frame_width: int,
    frame_height: int,
    *,
    min_gap_slots: int = 20,
    min_run_intervals: int = 3,
    max_velocity_deviation: float = 0.01,
) -> list[tuple[int, int]]:
    """Find long, consecutive anchor intervals with a stable velocity vector.

    Short intervals and changes in map reflection break a run.  Every accepted
    interval belongs to a locally stable window of ``min_run_intervals`` gaps.
    The normalized vector test checks both speed and direction; a near-zero
    speed cannot establish a motion model.  This remains an opt-in straight-leg
    assumption, never a general curvature guarantee.
    """
    if min_gap_slots < 1 or min_run_intervals < 2 or max_velocity_deviation <= 0:
        raise ValueError("Invalid straight-leg motion model thresholds")
    ids = sorted(int(fid) for fid in anchor_affines)
    if len(ids) < min_run_intervals + 1:
        return []
    center_px = np.array([frame_width / 2.0, frame_height / 2.0])
    centers: dict[int, np.ndarray] = {}
    signs: dict[int, bool] = {}
    for fid in ids:
        affine = np.asarray(anchor_affines[fid], dtype=np.float64)
        if affine.shape != (2, 3) or not np.all(np.isfinite(affine)):
            return []
        determinant = float(np.linalg.det(affine[:, :2]))
        if abs(determinant) <= 1e-12:
            return []
        centers[fid] = affine[:, :2] @ center_px + affine[:, 2]
        signs[fid] = determinant < 0

    runs: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for a, b in zip(ids, ids[1:]):
        if b - a >= min_gap_slots and signs[a] == signs[b]:
            current.append((a, b))
        else:
            if current:
                runs.append(current)
                current = []
    if current:
        runs.append(current)

    selected: set[tuple[int, int]] = set()
    for run in runs:
        if len(run) < min_run_intervals:
            continue
        velocities = np.asarray([(centers[b] - centers[a]) / (b - a) for a, b in run])
        for start in range(len(run) - min_run_intervals + 1):
            window = velocities[start : start + min_run_intervals]
            reference = np.median(window, axis=0)
            speed = float(np.linalg.norm(reference))
            if speed <= 1e-9:
                continue
            deviation = np.linalg.norm(window - reference, axis=1) / speed
            if np.all(deviation <= max_velocity_deviation):
                selected.update(run[start : start + min_run_intervals])
    return sorted(selected)


def interpolate_linear_anchor_intervals(
    anchor_affines: Mapping[int, np.ndarray],
    intervals: list[tuple[int, int]],
    frame_width: int,
    frame_height: int,
) -> dict[int, np.ndarray]:
    """Return exact endpoint anchors and interpolated interiors for selected gaps."""
    cx, cy = frame_width / 2.0, frame_height / 2.0
    output: dict[int, np.ndarray] = {}
    for a, b in intervals:
        left = np.asarray(anchor_affines[a], dtype=np.float64)
        right = np.asarray(anchor_affines[b], dtype=np.float64)
        sign = -1.0 if np.linalg.det(left[:, :2]) < 0 else 1.0
        if (np.linalg.det(right[:, :2]) < 0) != (sign < 0):
            continue
        output[a] = left.copy()
        output[b] = right.copy()
        start = _affine_to_state(left, cx, cy)
        end = _affine_to_state(right, cx, cy)
        delta = end - start
        delta[4] = np.arctan2(np.sin(delta[4]), np.cos(delta[4]))
        for fid in range(a + 1, b):
            state = start + (fid - a) / (b - a) * delta
            output[fid] = _state_to_affine(state, cx, cy, sign)
    return output
