"""Re-anchor blending of the reported position (vision-only plan 2.2).

Optical flow drifts between keyframes; when the next keyframe fix lands the
reported position used to step back by the whole accumulated drift (route3min
2026-11-10: every jump > 10 m in 12 runs is such a re-anchor, 11-28 m, while
the keyframe fixes themselves are accurate). The user's requirement is error
< 10 m without jumps, so the output is blended instead:

* a keyframe fix that differs from the continued output by more than
  ``min_m`` (and at most ``max_offset_m``) is absorbed into an output offset;
* every output then moves the offset toward zero by at most ``max_step_m``;
* a larger disagreement is a real relocation and is reported as is.

The tracker state (Kalman, optical-flow anchor) always uses the fix itself;
only the reported position is continuous. Positions are in local metres around
the first output (equirectangular, fine over a flight).
"""

from __future__ import annotations

import math

_R = 6_371_008.8


class OutputBlend:
    def __init__(
        self,
        max_step_m: float,
        min_m: float = 5.0,
        max_offset_m: float = 40.0,
        max_gap_s: float = 0.5,
    ) -> None:
        self.max_gap = float(max_gap_s)
        self.max_step = float(max_step_m)
        self.min_m = float(min_m)
        self.max_offset = float(max_offset_m)
        self.reset()

    def reset(self) -> None:
        self._origin: tuple[float, float] | None = None
        self._last: tuple[float, float] | None = None  # last output, local m
        self._last_t: float | None = None
        self._vel = (0.0, 0.0)
        self._off = (0.0, 0.0)

    # ── local plane ──
    def _xy(self, lat: float, lon: float) -> tuple[float, float]:
        lat0, lon0 = self._origin
        k = math.radians(1.0) * _R
        return (lon - lon0) * k * math.cos(math.radians(lat0)), (lat - lat0) * k

    def _ll(self, x: float, y: float) -> tuple[float, float]:
        lat0, lon0 = self._origin
        k = math.radians(1.0) * _R
        return lat0 + y / k, lon0 + x / (k * math.cos(math.radians(lat0)))

    def _decay(self) -> None:
        ox, oy = self._off
        n = math.hypot(ox, oy)
        if n <= self.max_step:
            self._off = (0.0, 0.0)
        else:
            f = (n - self.max_step) / n
            self._off = (ox * f, oy * f)

    def apply(
        self, lat: float, lon: float, t: float, is_keyframe: bool, absorb: bool = True
    ) -> tuple[float, float, float]:
        """(output lat, output lon, offset m) for a tracker position at time ``t``.

        ``absorb=False``: the previous outputs were not map-confirmed (dead
        reckoning) — a keyframe fix then replaces them at once instead of
        keeping their drift on screen (combined2 2026-11-11, wayback_2018 at
        111 s: an 8 m fix shown as 53 m after a dead-reckoning stretch).
        """
        if self._origin is None:
            self._origin = (lat, lon)
        x, y = self._xy(lat, lon)
        if self._last is None or self._last_t is None or not (0.0 < t - self._last_t <= 3.0):
            self._off = (0.0, 0.0)
            self._vel = (0.0, 0.0)
        else:
            dt = t - self._last_t
            px = self._last[0] + self._vel[0] * dt
            py = self._last[1] + self._vel[1] * dt
            ex = x + self._off[0] - px
            ey = y + self._off[1] - py
            e = math.hypot(ex, ey)
            if is_keyframe and absorb and self.min_m < e <= self.max_offset and dt <= self.max_gap:
                self._off = (self._off[0] - ex, self._off[1] - ey)
            elif e > self.max_offset or (is_keyframe and (dt > self.max_gap or not absorb)):
                self._off = (0.0, 0.0)  # relocation: report it, don't hide it
            self._decay()
        ox, oy = x + self._off[0], y + self._off[1]
        if self._last is not None and self._last_t is not None and 0.0 < t - self._last_t <= 3.0:
            dt = t - self._last_t
            vx, vy = (ox - self._last[0]) / dt, (oy - self._last[1]) / dt
            self._vel = (0.5 * self._vel[0] + 0.5 * vx, 0.5 * self._vel[1] + 0.5 * vy)
        self._last, self._last_t = (ox, oy), t
        olat, olon = self._ll(ox, oy)
        return olat, olon, math.hypot(*self._off)
