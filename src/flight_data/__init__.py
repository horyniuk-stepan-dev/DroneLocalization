"""Optional flight data (drone telemetry) for localization.

The system must work WITHOUT it — that is the default (``flight_data.source =
"none"``). When a stream is available it is used only as a PRIOR:

* heading → the rotation prior of the next keyframe (``yaw_hint_deg`` of
  ``Localizer.localize_frame``). A wrong heading costs one retrieval pass and
  falls back to the vision-only scan (rotation_rescan_min_score).
* altitude (AGL / MSL), attitude and GNSS are carried in ``FlightSample`` for the
  scale prior and the terrain work; nothing consumes them yet.

Sources implement ``FlightDataSource.sample_at(time_s)``. Shipped:
``NullFlightData`` and ``CsvFlightData`` (a time-indexed log; preset
"flightsim" reads FlightSimulator's telemetry.csv). A live stream (e.g. MAVLink
ATTITUDE / GLOBAL_POSITION_INT / VFR_HUD) only needs another ``sample_at``.

Conventions
-----------
heading_deg: compass, clockwise from north, of the direction the TOP of the
image points to (for a nadir camera whose image top is the nose: the drone
heading; otherwise add ``camera_yaw_offset_deg``).
yaw hint: the angle to rotate the query frame by (cv2 / np.rot90 sense,
positive = counter-clockwise on screen) so it matches reference frames whose
image top pointed to ``reference_heading_deg`` (0 for north-up / heading-hold
layers): ``hint = reference_heading - heading`` (mod 360). Example: drone
flying east (90) over north-up layers → hint 270 = rotate 90° clockwise.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class FlightSample:
    time_s: float
    heading_deg: float | None = None
    alt_agl_m: float | None = None
    alt_msl_m: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    lat: float | None = None
    lon: float | None = None


class FlightDataSource(Protocol):
    def sample_at(self, time_s: float) -> FlightSample | None: ...


class NullFlightData:
    """No telemetry: every query returns None (vision-only localization)."""

    def sample_at(self, time_s: float) -> FlightSample | None:
        return None


def heading_to_yaw_hint(
    heading_deg: float, reference_heading_deg: float = 0.0, camera_yaw_offset_deg: float = 0.0
) -> float:
    """Rotation (cv2 sense, degrees in [0, 360)) aligning the query to the references."""
    return (
        float(reference_heading_deg) - float(heading_deg) - float(camera_yaw_offset_deg)
    ) % 360.0


# Column mapping per preset: field -> (column, transform).
_PRESETS: dict[str, dict[str, tuple[str, object]]] = {
    "generic": {
        "time_s": ("time_s", float),
        "heading_deg": ("heading_deg", float),
        "alt_agl_m": ("alt_agl_m", float),
        "alt_msl_m": ("alt_msl_m", float),
        "pitch_deg": ("pitch_deg", float),
        "roll_deg": ("roll_deg", float),
        "lat": ("lat", float),
        "lon": ("lon", float),
    },
    # FlightSimulator telemetry.csv: yaw_rad is counter-clockwise from north
    # (AutoPilot: atan2(-vx, vy)), alt_z is height above the simulator base plane.
    "flightsim": {
        "time_s": ("timestamp", float),
        "heading_deg": ("yaw_rad", lambda v: (-math.degrees(float(v))) % 360.0),
        "alt_msl_m": ("alt_z", float),
    },
}


@dataclass
class CsvFlightData:
    """Time-indexed telemetry log with linear interpolation (heading on the circle).

    ``time_offset_s`` maps video time to log time (log = video + offset). A query
    farther than ``max_gap_s`` from the nearest row returns None (stale data is
    worse than none).
    """

    path: str | Path
    preset: str = "generic"
    time_offset_s: float = 0.0
    max_gap_s: float = 1.0
    _t: np.ndarray = field(init=False, repr=False)
    _cols: dict[str, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.preset not in _PRESETS:
            raise ValueError(f"unknown flight data preset {self.preset!r}")
        mapping = _PRESETS[self.preset]
        rows: dict[str, list[float]] = {k: [] for k in mapping}
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = set(reader.fieldnames or [])
            time_col = mapping["time_s"][0]
            if time_col not in header:
                raise ValueError(f"{self.path}: no time column {time_col!r}")
            present = {k: v for k, v in mapping.items() if v[0] in header}
            for row in reader:
                for key, (col, fn) in present.items():
                    raw = row.get(col, "")
                    rows[key].append(fn(raw) if raw not in ("", None) else np.nan)
        order = np.argsort(np.asarray(rows["time_s"], dtype=np.float64), kind="stable")
        self._t = np.asarray(rows["time_s"], dtype=np.float64)[order]
        self._cols = {
            k: np.asarray(v, dtype=np.float64)[order]
            for k, v in rows.items()
            if k != "time_s" and len(v) == len(order)
        }
        if len(self._t) == 0:
            raise ValueError(f"{self.path}: no rows")

    def sample_at(self, time_s: float) -> FlightSample | None:
        t = float(time_s) + float(self.time_offset_s)
        i = int(np.searchsorted(self._t, t))
        lo, hi = max(0, i - 1), min(len(self._t) - 1, i)
        if min(abs(self._t[lo] - t), abs(self._t[hi] - t)) > self.max_gap_s:
            return None
        w = (
            0.0
            if hi == lo
            else float(np.clip((t - self._t[lo]) / (self._t[hi] - self._t[lo]), 0, 1))
        )
        values: dict[str, float | None] = {}
        for key, col in self._cols.items():
            a, b = col[lo], col[hi]
            if not (np.isfinite(a) and np.isfinite(b)):
                v = a if np.isfinite(a) else b
                values[key] = float(v) if np.isfinite(v) else None
            elif key == "heading_deg":
                d = ((b - a + 180.0) % 360.0) - 180.0
                values[key] = float((a + w * d) % 360.0)
            else:
                values[key] = float(a + w * (b - a))
        return FlightSample(time_s=t, **values)


class FlightPrior:
    """Turns ``flight_data.*`` config + a source into per-keyframe priors."""

    def __init__(self, source: FlightDataSource, config) -> None:
        from config import get_cfg

        self.source = source
        self.use_heading = bool(get_cfg(config, "flight_data.use_heading", True))
        self.reference_heading_deg = float(
            get_cfg(config, "flight_data.reference_heading_deg", 0.0)
        )
        self.camera_yaw_offset_deg = float(
            get_cfg(config, "flight_data.camera_yaw_offset_deg", 0.0)
        )

    def yaw_hint_deg(self, video_time_s: float) -> float | None:
        if not self.use_heading:
            return None
        sample = self.source.sample_at(video_time_s)
        if sample is None or sample.heading_deg is None or not math.isfinite(sample.heading_deg):
            return None
        return heading_to_yaw_hint(
            sample.heading_deg, self.reference_heading_deg, self.camera_yaw_offset_deg
        )


def build_flight_prior(config) -> FlightPrior | None:
    """``None`` when ``flight_data.source == "none"`` (the default)."""
    from config import get_cfg

    kind = str(get_cfg(config, "flight_data.source", "none"))
    if kind == "none":
        return None
    if kind == "csv":
        source = CsvFlightData(
            path=str(get_cfg(config, "flight_data.csv_path", "")),
            preset=str(get_cfg(config, "flight_data.csv_preset", "generic")),
            time_offset_s=float(get_cfg(config, "flight_data.time_offset_s", 0.0)),
            max_gap_s=float(get_cfg(config, "flight_data.max_gap_s", 1.0)),
        )
        return FlightPrior(source, config)
    raise ValueError(f"unknown flight_data.source {kind!r}")
