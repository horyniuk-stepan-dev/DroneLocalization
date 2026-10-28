"""Keyframe-anchored Lucas-Kanade optical flow (shared by worker and replay).

Extracted verbatim from TrackingWorker.run (2026-10-27) so the offline replay
(scripts/replay_tracking_ground_truth.py) exercises exactly the OF path the
live worker runs. LK always tracks FROM the keyframe: the anchor image and
points are set once per accepted keyframe and never advanced on OF frames, so
the displacement grows with time since the keyframe and errors do not chain.

Pure OpenCV/numpy — no Qt, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class FlowMeasurement:
    dx_px: float
    dy_px: float
    flow_affine: np.ndarray | None
    flow_quality: float | None
    n_points: int


class KeyframeFlow:
    """LK tracker anchored at the last accepted keyframe."""

    def __init__(
        self,
        fb_check: bool = False,
        fb_max_px: float = 2.0,
        half_res: bool = False,
        win_px: int = 15,
        max_level: int = 2,
        max_anchor_age_s: float = 0.0,
    ):
        self.fb_check = bool(fb_check)
        self.fb_max_px = float(fb_max_px)
        self.half_res = bool(half_res)
        # LK capture range grows ~2^max_level * win/2. Legacy 15/2 covers ~60 px;
        # LK always tracks FROM the keyframe, so at 500 m AGL / 50 m/s the scene
        # moves ~160 px by the 25th frame and the track is lost (measured
        # 2026-10-28 on route3min: 37-74 px error at offsets 15-25 vs ~1 px
        # with 21/4).
        self.win = (max(int(win_px), 5),) * 2
        self.max_level = max(int(max_level), 0)
        # Anchor age limit (0 = off, legacy). The anchor only moves on an
        # accepted keyframe; after a FAILED keyframe LK keeps tracking from a
        # frame 1-2 s old, far beyond its capture range (route3min 2026-10-28:
        # raw OF 235-435 m off). With a limit the stale anchor is dropped and
        # the next frame is a keyframe attempt, like after a lost track.
        self.max_anchor_age_s = max(float(max_anchor_age_s), 0.0)
        self.anchor_t: float | None = None
        self.prev_gray: np.ndarray | None = None
        self.prev_gray_half: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None

    @property
    def has_anchor(self) -> bool:
        """Mirrors the worker's ``prev_pts_for_of is not None`` test."""
        return self.prev_pts is not None

    def clear(self) -> None:
        self.prev_pts = None

    def expire(self, t: float | None) -> bool:
        """Drop the anchor when it is older than max_anchor_age_s. Call before
        deciding keyframe vs OF for frame time ``t``; True if it was dropped."""
        if (
            self.max_anchor_age_s <= 0.0
            or t is None
            or self.anchor_t is None
            or self.prev_pts is None
        ):
            return False
        if t - self.anchor_t > self.max_anchor_age_s:
            self.prev_pts = None
            return True
        return False

    def anchor(self, gray: np.ndarray, t: float | None = None) -> None:
        """Set a new anchor after an accepted (non retrieval-only) keyframe."""
        self.anchor_t = t
        self.prev_gray = gray
        self.prev_gray_half = (
            cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            if self.half_res
            else None
        )
        self.prev_pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=200, qualityLevel=0.01, minDistance=30, mask=None
        )

    def track(self, curr_gray: np.ndarray) -> FlowMeasurement | None:
        """Flow from the anchor to ``curr_gray``.

        Returns None and drops the anchor when too few points survive — the
        worker then treats the next frame as a keyframe (same as before).
        """
        if self.prev_pts is None or len(self.prev_pts) <= 10:
            self.prev_pts = None
            return None

        of_scale = 0.5 if (self.half_res and self.prev_gray_half is not None) else 1.0
        if of_scale != 1.0:
            g_prev = self.prev_gray_half
            g_curr = cv2.resize(curr_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            pts_prev = np.ascontiguousarray(self.prev_pts * of_scale, dtype=np.float32)
        else:
            g_prev, g_curr, pts_prev = self.prev_gray, curr_gray, self.prev_pts

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            g_prev,
            g_curr,
            pts_prev,
            None,
            winSize=self.win,
            maxLevel=self.max_level,
        )
        keep = status.reshape(-1) == 1

        if self.fb_check and keep.any():
            back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                g_curr,
                g_prev,
                curr_pts,
                None,
                winSize=self.win,
                maxLevel=self.max_level,
            )
            rt_err = np.linalg.norm(back_pts.reshape(-1, 2) - pts_prev.reshape(-1, 2), axis=1)
            fb_ok = (back_status.reshape(-1) == 1) & (rt_err <= self.fb_max_px)
            if int((keep & fb_ok).sum()) >= 10:
                keep = keep & fb_ok

        good_new = curr_pts.reshape(-1, 2)[keep]
        good_old = pts_prev.reshape(-1, 2)[keep]
        if of_scale != 1.0:
            good_new = good_new / of_scale
            good_old = good_old / of_scale

        if len(good_new) <= 10:
            self.prev_pts = None
            return None

        flow_vectors = good_new - good_old
        dx_px, dy_px = np.median(flow_vectors, axis=0)

        flow_affine = None
        flow_quality = None
        try:
            S_of, of_mask = cv2.estimateAffinePartial2D(
                good_old,
                good_new,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )
            if S_of is not None and np.all(np.isfinite(S_of)):
                flow_affine = S_of
                if of_mask is not None and len(of_mask) > 0:
                    inlier_ratio = float(of_mask.sum()) / len(of_mask)
                    n_norm = min(1.0, len(good_new) / 120.0)
                    flow_quality = inlier_ratio * n_norm
        except cv2.error as e:
            logger.debug(f"OF affine estimation failed: {e}")

        return FlowMeasurement(
            dx_px=dx_px,
            dy_px=dy_px,
            flow_affine=flow_affine,
            flow_quality=flow_quality,
            n_points=int(len(good_new)),
        )
