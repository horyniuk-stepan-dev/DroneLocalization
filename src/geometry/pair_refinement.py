"""Offline pair refinement and quality gates in original image pixels."""

import hashlib
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.geometry.frame_transform import project_points, valid_footprint
from src.geometry.point_spread import inlier_spread


def symmetric_errors(H, first, second):
    forward = project_points(H, first) - second
    backward = project_points(np.linalg.inv(H), second) - first
    return np.sqrt((np.sum(forward**2, axis=1) + np.sum(backward**2, axis=1)) / 2)


def refine_pair(H, first, second, width, height):
    """Robust symmetric refit after MAD, retaining the seed if it gets worse."""
    size = float(max(width, height))
    N = np.array(
        [[1 / size, 0, -width / (2 * size)], [0, 1 / size, -height / (2 * size)], [0, 0, 1.0]]
    )
    inv_N = np.linalg.inv(N)
    p, q = project_points(N, first), project_points(N, second)
    normalized = N @ H @ inv_N
    normalized /= normalized[2, 2]

    def unpack(x):
        return np.r_[x, 1.0].reshape(3, 3)

    def residual(x):
        matrix = unpack(x)
        try:
            return (
                np.column_stack(
                    [project_points(matrix, p) - q, project_points(np.linalg.inv(matrix), q) - p]
                ).ravel()
                * size
            )
        except (ValueError, np.linalg.LinAlgError):
            return np.full(4 * len(p), 1e6)

    before = residual(normalized.ravel()[:8])
    fit = least_squares(residual, normalized.ravel()[:8], loss="soft_l1", f_scale=1.0, max_nfev=40)
    candidate = inv_N @ unpack(fit.x) @ N
    after = residual(fit.x)

    def cost(r):
        return np.sum(np.sqrt(1 + r * r) - 1)

    if (
        valid_footprint(candidate, width, height, orientation=1)
        and cost(after) <= cost(before) + 1e-8
    ):
        return candidate / candidate[2, 2]
    return H


def pair_quality(
    H,
    first,
    second,
    total_matches,
    width,
    height,
    *,
    min_ratio=0.3,
    min_spread=0.025,
    max_error=3.0,
    inlier_count=None,
):
    """Shared temporal/spatial acceptance and auditable diagnostics."""
    if not valid_footprint(H, width, height, orientation=1):
        return False, {"rejection": "invalid_footprint"}
    if not valid_footprint(np.linalg.inv(H), width, height, orientation=1):
        return False, {"rejection": "invalid_inverse_footprint"}
    error = symmetric_errors(H, first, second)
    spread = min(
        inlier_spread(first, width, height, principal=True) or 0.0,
        inlier_spread(second, width, height, principal=True) or 0.0,
    )
    coverage, extrapolation = [], []
    for points in (first, second):
        hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
        coverage.append(float(cv2.contourArea(hull) / (width * height)))
        extrapolation.append(
            max(0.0, -cv2.pointPolygonTest(hull, (width / 2, height / 2), True))
            / np.hypot(width, height)
        )
    report = {
        "inliers": len(first) if inlier_count is None else inlier_count,
        "inlier_ratio": (len(first) if inlier_count is None else inlier_count)
        / max(total_matches, 1),
        "spread": spread,
        "hull_coverage": min(coverage),
        "center_extrapolation": max(extrapolation),
        "symmetric_rms_px": float(np.sqrt(np.mean(error**2))),
        "symmetric_p95_px": float(np.percentile(error, 95)),
    }
    reason = None
    if report["inlier_ratio"] < min_ratio:
        reason = "inlier_ratio"
    elif spread < min_spread or min(coverage) < 0.002:
        reason = "point_support"
    elif report["symmetric_p95_px"] > max_error or not np.isfinite(error).all():
        reason = "symmetric_error"
    elif max(extrapolation) > 0.5:
        reason = "center_extrapolation"
    report["rejection"] = reason
    return reason is None, report


def refine_subpixel(first_image, second_image, first, second, max_correction=2.0):
    """LK initialized by matched observations, with forward/backward validation.

    Keep the original measurement when LK is unsupported; a failed photometric
    refinement must not discard valid wide-baseline learned correspondences.
    """
    p = np.asarray(first, dtype=np.float32).reshape(-1, 1, 2)
    q = np.asarray(second, dtype=np.float32).reshape(-1, 1, 2)
    if len(p) == 0:
        return np.asarray(second).copy(), 0
    options = dict(
        winSize=(21, 21),
        maxLevel=1,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.001),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        minEigThreshold=1e-4,
    )
    updated, forward, _ = cv2.calcOpticalFlowPyrLK(
        first_image, second_image, p, q.copy(), **options
    )
    if updated is None:
        return np.asarray(second).copy(), 0
    back, backward, _ = cv2.calcOpticalFlowPyrLK(
        second_image, first_image, updated, p.copy(), **options
    )
    if back is None:
        return np.asarray(second).copy(), 0
    new = updated.reshape(-1, 2)
    good = (
        forward.ravel().astype(bool)
        & backward.ravel().astype(bool)
        & (np.linalg.norm(back.reshape(-1, 2) - first, axis=1) < 0.5)
        & (np.linalg.norm(new - second, axis=1) <= max_correction)
        & np.isfinite(new).all(axis=1)
    )
    h, w = second_image.shape[:2]
    good &= (new[:, 0] >= 10) & (new[:, 0] < w - 10) & (new[:, 1] >= 10) & (new[:, 1] < h - 10)
    result = np.asarray(second, dtype=np.float64).copy()
    result[good] = new[good]
    return result, int(good.sum())


class SourceFrames:
    """Bounded source-frame cache; verify video identity before refining points."""

    def __init__(self, metadata, check_running=lambda: None, capacity=8):
        self.check_running = check_running
        self.capacity = capacity
        self.cache = OrderedDict()
        self.cap = None
        self.step = int(metadata.get("frame_step", 0))
        self.width = int(metadata.get("frame_width", 0))
        self.height = int(metadata.get("frame_height", 0))
        path = Path(metadata.get("source_path", ""))
        expected = metadata.get("source_sha256", "")
        if not path.is_file() or not expected or min(self.step, self.width, self.height) <= 0:
            raise ValueError(
                "Source video identity unavailable; rebuild database to enable subpixel refinement"
            )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                check_running()
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("Source video changed since database build")
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.close()
            raise ValueError("Cannot open source video")

    def get(self, frame_id):
        self.check_running()
        if frame_id in self.cache:
            self.cache.move_to_end(frame_id)
            return self.cache[frame_id]
        if self.cap is None or not self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id * self.step):
            raise ValueError(f"Cannot seek source frame {frame_id}")
        ok, frame = self.cap.read()
        if not ok or frame.shape[:2] != (self.height, self.width):
            raise ValueError(f"Source frame {frame_id} is missing or has wrong dimensions")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.cache[frame_id] = gray
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return gray

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cache.clear()
