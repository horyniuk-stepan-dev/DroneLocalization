"""Vision-only altitude (query ground pixel size) between and across keyframes.

Vision-only plan 2.1 (``localization.altitude_hint.visual_track``, default off).
The altitude hint without telemetry uses the last fix's scale prior, but every
failed keyframe invalidates that prior — so the hint disappears exactly when
the drone has drifted between layers and needs it. This tracker keeps a query
ground-pixel-size estimate alive:

* keyframe fix: query GSD = scale prior (query / layer) x that layer's GSD;
* optical flow: the KF -> current similarity's scale s tracks the altitude
  change (descending enlarges the ground: GSD_now = GSD_kf / s);
* the dead-reckoning chain moves its anchor: the base follows the rebase;
* the estimate expires ``max_age_s`` after the last keyframe fix.

Assumes a near-nadir view over locally flat ground (scale = altitude ratio).
"""

from __future__ import annotations

import math

import numpy as np

# A single KF -> current similarity outside this range is a tracking error,
# not an altitude change within one keyframe interval or chain link.
_MIN_STEP, _MAX_STEP = 0.5, 2.0


def flow_scale(flow_affine) -> float | None:
    """Isotropic scale of a 2x3 similarity/affine (sqrt |det|), None if unusable."""
    try:
        a = np.asarray(flow_affine, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if a.shape != (2, 3) or not np.all(np.isfinite(a)):
        return None
    det = abs(float(np.linalg.det(a[:, :2])))
    if det <= 0.0:
        return None
    s = math.sqrt(det)
    return s if _MIN_STEP <= s <= _MAX_STEP else None


class VisualScaleTrack:
    def __init__(self, max_age_s: float = 60.0) -> None:
        self.max_age_s = float(max_age_s)
        self.reset()

    def reset(self) -> None:
        self._base: float | None = None  # query GSD at the current flow anchor
        self._current: float | None = None
        self._fix_t: float | None = None

    def on_keyframe(self, t: float | None, query_gsd: float | None) -> None:
        if query_gsd is None or not math.isfinite(query_gsd) or query_gsd <= 0:
            return
        self._base = self._current = float(query_gsd)
        self._fix_t = None if t is None else float(t)

    def on_flow(self, flow_affine) -> None:
        """OF measurement from the current anchor to this frame."""
        s = flow_scale(flow_affine)
        if self._base is not None and s is not None:
            self._current = self._base / s

    def on_rebase(self, flow_affine) -> None:
        """The dead-reckoning chain moved the anchor by ``flow_affine``."""
        s = flow_scale(flow_affine)
        if self._base is not None and s is not None:
            self._base = self._current = self._base / s

    def estimate(self, t: float | None) -> float | None:
        if self._current is None:
            return None
        if t is not None and self._fix_t is not None and t - self._fix_t > self.max_age_s:
            return None
        return self._current
