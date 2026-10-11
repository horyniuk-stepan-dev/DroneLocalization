"""Spatial prior: map frames near the dead-reckoned position after a lost fix.

The temporal prior (``Localizer._try_temporal_prior``) probes the last matched
map frame and its neighbours by DB index. Two things break it (diagnosed
2026-10-10 on the wb2019 route, other-satellite imagery):

- index neighbours are neighbours along the map's recording path, not on the
  ground: when the drone leaves that strip, the frame it now sees can be dozens
  of indices away;
- a lost frame's in-frame retry clears the angle prior, and the temporal prior
  needs an angle, so it stays off until the next fix (retrieval only).

This class keeps the last two confirmed fixes, predicts the position at
constant velocity, and lists the map frames nearest to the prediction. The
localizer uses it as a second stage when the index stage finds nothing. It
holds no models and no I/O: positions are WGS84 degrees, distances are an
equirectangular approximation (exact enough over a few kilometres).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np

from config import get_cfg

EARTH_RADIUS_M = 6371008.8
# GeoreferenceStatus.SUPPORTED (src/geometry/calibration_provenance.py); only a
# pre-filter here, the caller's `usable` check is authoritative.
_GEOREF_SUPPORTED = 1


class SpatialPrior:
    """Dead reckoning from the last confirmed fixes plus a nearest-frame query."""

    def __init__(
        self,
        enabled: bool = False,
        max_age_s: float = 30.0,
        nearest_k: int = 4,
        radius_m: float = 1500.0,
        verify: int = 2,
        max_speed_mps: float = 120.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_age_s = float(max_age_s)
        self.nearest_k = max(0, int(nearest_k))
        self.radius_m = float(radius_m)
        self.verify = max(1, int(verify))
        self.max_speed_mps = float(max_speed_mps)
        self.reset()

    @classmethod
    def from_config(cls, config: dict, enabled: bool = True) -> SpatialPrior:
        return cls(
            enabled=enabled and bool(get_cfg(config, "localization.spatial_prior", False)),
            max_age_s=float(get_cfg(config, "localization.spatial_prior_max_age_s", 30.0)),
            nearest_k=int(get_cfg(config, "localization.spatial_prior_nearest_k", 4)),
            radius_m=float(get_cfg(config, "localization.spatial_prior_radius_m", 1500.0)),
            verify=int(get_cfg(config, "localization.spatial_prior_verify", 2)),
            max_speed_mps=float(get_cfg(config, "tracking.max_speed_mps", 120.0)),
        )

    def reset(self) -> None:
        self._clock: float | None = None
        self._fixes: list[tuple[float, float, float]] = []  # (t, lat, lon), last two
        self.angle: float | None = None
        self.scale: float | None = None

    # ── time ────────────────────────────────────────────────────────────────
    def tick(self, timestamp: float | None, dt: float | None = 1.0) -> float:
        """Current time of the frame being localized (seconds).

        The caller's timestamp when it has one (video time in the worker and the
        replay), otherwise the previous time + dt. Time running backwards (a new
        video, a seek) drops the stored fixes.
        """
        if timestamp is not None and math.isfinite(float(timestamp)):
            now = float(timestamp)
        else:
            step = float(dt) if dt is not None and math.isfinite(float(dt)) else 0.0
            now = (self._clock or 0.0) + max(0.0, step)
        if self._fixes and now < self._fixes[-1][0]:
            self._fixes.clear()
            self.angle = self.scale = None
        self._clock = now
        return now

    def record_fix(
        self,
        now: float,
        lat: float,
        lon: float,
        angle: float | None = None,
        scale: float | None = None,
    ) -> None:
        """Remember a confirmed fix, the angle prior and the scale prior after it."""
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return
        if self._fixes and now <= self._fixes[-1][0]:
            self._fixes[-1] = (float(now), float(lat), float(lon))
        else:
            self._fixes = [*self._fixes[-1:], (float(now), float(lat), float(lon))]
        if angle is not None:
            self.angle = float(angle)
        if scale is not None:
            self.scale = float(scale)

    def age(self, now: float) -> float | None:
        """Seconds since the last fix, or None without one."""
        return now - self._fixes[-1][0] if self._fixes else None

    def predict(self, now: float) -> tuple[float, float] | None:
        """Constant-velocity position at `now`, or None (no fix / fix too old).

        Velocity from the last two fixes; zero when they are further apart than
        max_age_s or imply a speed above max_speed_mps (a wrong fix).
        """
        if not self.enabled or not self._fixes:
            return None
        t2, lat2, lon2 = self._fixes[-1]
        age = now - t2
        if age < 0 or age > self.max_age_s:
            return None
        vlat = vlon = 0.0
        if len(self._fixes) == 2:
            t1, lat1, lon1 = self._fixes[0]
            span = t2 - t1
            if 0.0 < span <= self.max_age_s:
                dy, dx = _offset_m(lat1, lon1, lat2, lon2)
                if math.hypot(dx, dy) / span <= self.max_speed_mps:
                    vlat = (lat2 - lat1) / span
                    vlon = (lon2 - lon1) / span
        return lat2 + vlat * age, lon2 + vlon * age

    # ── map frames ──────────────────────────────────────────────────────────
    def nearest_ids(
        self,
        database,
        lat: float,
        lon: float,
        exclude: Iterable[int] = (),
        usable: Callable[[int], bool] | None = None,
    ) -> list[int]:
        """Up to nearest_k frames of `database` within radius_m of (lat, lon).

        Ordered by distance. `frame_gps` (N × [lat, lon]) is the map's per-frame
        position; frames without one, outside the radius, in `exclude`, or
        rejected by `usable` (no calibration / georeference) are skipped.
        """
        gps = getattr(database, "frame_gps", None)
        if gps is None or self.nearest_k == 0:
            return []
        gps = np.asarray(gps, dtype=np.float64)
        if gps.ndim != 2 or gps.shape[0] == 0 or gps.shape[1] < 2:
            return []
        dy, dx = _offset_m(lat, lon, gps[:, 0], gps[:, 1])
        d = np.hypot(dx, dy)
        d[~np.isfinite(d)] = np.inf
        d[d > self.radius_m] = np.inf
        status = getattr(database, "frame_georef_status", None)
        if status is not None and len(status) == len(d):
            d[np.asarray(status) != _GEOREF_SUPPORTED] = np.inf
        n_in = int(np.count_nonzero(np.isfinite(d)))
        if n_in == 0:
            return []
        order = np.argsort(d, kind="stable")[:n_in]
        skip = {int(c) for c in exclude}
        out: list[int] = []
        for c in order[: 50 * self.nearest_k]:
            c = int(c)
            if c in skip or (usable is not None and not usable(c)):
                continue
            out.append(c)
            if len(out) >= self.nearest_k:
                break
        return out


def _offset_m(lat0, lon0, lat, lon):
    """(north, east) metres from (lat0, lon0) to (lat, lon), equirectangular."""
    k = math.cos(math.radians(float(lat0)))
    dy = np.radians(np.asarray(lat) - lat0) * EARTH_RADIUS_M
    dx = np.radians(np.asarray(lon) - lon0) * EARTH_RADIUS_M * k
    return dy, dx
