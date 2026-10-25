"""Optional flight data (drone telemetry) for localization.

The system must work WITHOUT it — that is the default (``flight_data.source =
"none"``). When a stream is available it is used only as a PRIOR:

* heading → the rotation prior of the next keyframe (``yaw_hint_deg`` of
  ``Localizer.localize_frame``). A wrong heading costs one retrieval pass and
  falls back to the vision-only scan (rotation_rescan_min_score).
* height above ground (else above take-off) → ``localization.altitude_hint``
  (default off): the expected scale of the layers nearest that height is tried
  first. A rough value is enough; without it the hint uses the last fix's scale.
* MSL altitude, attitude and GNSS are carried in ``FlightSample`` for the terrain
  work; nothing consumes them yet.

Sources implement ``FlightDataSource.sample_at(time_s)``. Shipped:
``NullFlightData``, ``CsvFlightData`` (a time-indexed log; preset "flightsim"
reads FlightSimulator's telemetry.csv, "generic" also reads DroneSimUE's
telemetry.csv / frames.csv) and ``MavlinkFlightData`` (a live MAVLink stream:
ATTITUDE, GLOBAL_POSITION_INT incl. relative_alt, TERRAIN_REPORT / downward
DISTANCE_SENSOR).

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
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Protocol

import numpy as np


@dataclass(frozen=True)
class FlightSample:
    time_s: float
    heading_deg: float | None = None
    alt_agl_m: float | None = None
    alt_rel_m: float | None = None  # above take-off / home (baro), when AGL is unknown
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
        "alt_rel_m": ("alt_rel_m", float),
        "alt_msl_m": ("alt_msl_m", float),
        "pitch_deg": ("pitch_deg", float),
        "roll_deg": ("roll_deg", float),
        "lat": ("lat", float),
        "lon": ("lon", float),
    },
    # FlightSimulator telemetry.csv: yaw_rad is counter-clockwise from north
    # (AutoPilot: atan2(-vx, vy)), alt_z is height above the simulator base plane
    # (also the relative height for the altitude hint).
    "flightsim": {
        "time_s": ("timestamp", float),
        "heading_deg": ("yaw_rad", lambda v: (-math.degrees(float(v))) % 360.0),
        "alt_msl_m": ("alt_z", float),
        "alt_rel_m": ("alt_z", float),
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


_LIVE_FIELDS = (
    "heading_deg",
    "alt_agl_m",
    "alt_rel_m",
    "alt_msl_m",
    "pitch_deg",
    "roll_deg",
    "lat",
    "lon",
)
_MAV_TYPES = ["ATTITUDE", "GLOBAL_POSITION_INT", "TERRAIN_REPORT", "DISTANCE_SENSOR"]
_MAV_SENSOR_ROTATION_PITCH_270 = 25  # DISTANCE_SENSOR.orientation: facing down
_log = logging.getLogger(__name__)


class MavlinkFlightData:
    """Live telemetry from a MAVLink stream (DroneSimUE, or a flight controller
    behind a telemetry radio / OpenIPC link).

    A daemon thread keeps the last ``history_s`` seconds of state, stamped with
    the LOCAL receive time; the stream's own clock is not used. ``sample_at``
    ignores its argument's origin (live video time starts at the worker's start)
    and returns the state at ``now + time_offset_s``: use a negative offset to
    compensate the video latency (e.g. -0.15 when frames arrive 150 ms late).
    No message newer than ``max_gap_s`` around that time -> None.

    The UDP socket opens on the first query and closes with ``close()``; a new
    instance on the same URL closes the previous one (so a worker that exited
    early does not keep the port).
    """

    _active: ClassVar[dict[str, MavlinkFlightData]] = {}
    # Re-entrant: start() closes the previous owner while holding it.
    _active_lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(
        self,
        url: str = "udpin:0.0.0.0:14550",
        time_offset_s: float = 0.0,
        max_gap_s: float = 1.0,
        history_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        autostart: bool = True,
    ) -> None:
        self.url = url
        self.autostart = autostart
        self.time_offset_s = float(time_offset_s)
        self.max_gap_s = float(max_gap_s)
        self.history_s = float(history_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._state: dict[str, float] = {}
        self._history: deque[tuple[float, dict[str, float]]] = deque()
        self._last_attitude_t: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._conn = None

    # -- receiving ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            from pymavlink import mavutil
        except ImportError as e:  # optional dependency
            raise RuntimeError(
                "flight_data.source = mavlink needs pymavlink: python -m pip install pymavlink"
            ) from e
        with MavlinkFlightData._active_lock:
            previous = MavlinkFlightData._active.get(self.url)
            if previous is not None and previous is not self:
                previous.close()
            self._conn = mavutil.mavlink_connection(self.url, dialect="common")
            MavlinkFlightData._active[self.url] = self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="MavlinkFlightData", daemon=True)
        self._thread.start()
        _log.info("MAVLink flight data: listening on %s", self.url)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._conn.recv_match(type=_MAV_TYPES, blocking=True, timeout=0.2)
            except Exception as e:  # socket closed by close(), or a broken frame
                if self._stop.is_set():
                    break
                _log.debug("MAVLink receive error: %s", e)
                continue
            if msg is not None:
                self.ingest(msg.get_type(), msg.to_dict(), self._clock())

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        with MavlinkFlightData._active_lock:
            if MavlinkFlightData._active.get(self.url) is self:
                del MavlinkFlightData._active[self.url]

    def ingest(self, kind: str, fields: dict, t: float) -> None:
        """One decoded message (``msg.get_type()``, ``msg.to_dict()``) received at ``t``."""
        updates: dict[str, float] = {}
        if kind == "ATTITUDE":
            updates["heading_deg"] = math.degrees(float(fields["yaw"])) % 360.0
            updates["pitch_deg"] = math.degrees(float(fields["pitch"]))
            updates["roll_deg"] = math.degrees(float(fields["roll"]))
            self._last_attitude_t = t
        elif kind == "GLOBAL_POSITION_INT":
            updates["lat"] = float(fields["lat"]) / 1e7
            updates["lon"] = float(fields["lon"]) / 1e7
            updates["alt_msl_m"] = float(fields["alt"]) / 1000.0
            if fields.get("relative_alt") is not None:
                updates["alt_rel_m"] = float(fields["relative_alt"]) / 1000.0
            hdg = int(fields.get("hdg", 65535))
            attitude_fresh = self._last_attitude_t is not None and t - self._last_attitude_t <= 1.0
            if hdg != 65535 and not attitude_fresh:
                updates["heading_deg"] = (hdg / 100.0) % 360.0
        elif kind == "TERRAIN_REPORT":
            updates["alt_agl_m"] = float(fields["current_height"])
        elif kind == "DISTANCE_SENSOR":
            if int(fields.get("orientation", -1)) == _MAV_SENSOR_ROTATION_PITCH_270:
                updates["alt_agl_m"] = float(fields["current_distance"]) / 100.0
        if not updates:
            return
        with self._lock:
            self._state.update(updates)
            snapshot = (t, dict(self._state))
            if self._history and abs(t - self._history[-1][0]) < 1e-3:
                self._history[-1] = snapshot  # same packet burst: keep the complete state
            else:
                self._history.append(snapshot)
            while self._history and t - self._history[0][0] > self.history_s:
                self._history.popleft()

    # -- querying ------------------------------------------------------------
    def sample_at(self, time_s: float) -> FlightSample | None:
        if self.autostart and self._thread is None:
            self.start()
        target = self._clock() + self.time_offset_s
        with self._lock:
            history = list(self._history)
        if not history:
            return None
        times = [h[0] for h in history]
        i = int(np.searchsorted(times, target))
        lo, hi = max(0, i - 1), min(len(history) - 1, i)
        if min(abs(times[lo] - target), abs(times[hi] - target)) > self.max_gap_s:
            return None
        a, b = history[lo][1], history[hi][1]
        w = (
            0.0
            if hi == lo
            else float(np.clip((target - times[lo]) / (times[hi] - times[lo]), 0, 1))
        )
        values: dict[str, float | None] = {}
        for key in _LIVE_FIELDS:
            va, vb = a.get(key), b.get(key)
            if va is None or vb is None:
                values[key] = vb if vb is not None else va
            elif key == "heading_deg":
                d = ((vb - va + 180.0) % 360.0) - 180.0
                values[key] = (va + w * d) % 360.0
            else:
                values[key] = va + w * (vb - va)
        return FlightSample(time_s=float(time_s), **values)


class FlightPrior:
    """Turns ``flight_data.*`` config + a source into per-keyframe priors."""

    def __init__(self, source: FlightDataSource, config) -> None:
        from config import get_cfg

        self.source = source
        self.use_heading = bool(get_cfg(config, "flight_data.use_heading", True))
        self.use_altitude = bool(get_cfg(config, "flight_data.use_altitude", True))
        self.reference_heading_deg = float(
            get_cfg(config, "flight_data.reference_heading_deg", 0.0)
        )
        self.camera_yaw_offset_deg = float(
            get_cfg(config, "flight_data.camera_yaw_offset_deg", 0.0)
        )

    def close(self) -> None:
        """Releases a live source (MAVLink socket); no-op for logs."""
        close = getattr(self.source, "close", None)
        if callable(close):
            close()

    def yaw_hint_deg(self, video_time_s: float) -> float | None:
        if not self.use_heading:
            return None
        sample = self.source.sample_at(video_time_s)
        if sample is None or sample.heading_deg is None or not math.isfinite(sample.heading_deg):
            return None
        return heading_to_yaw_hint(
            sample.heading_deg, self.reference_heading_deg, self.camera_yaw_offset_deg
        )

    def altitude_m(self, video_time_s: float) -> float | None:
        """Height for ``localization.altitude_hint``: above ground, else above take-off."""
        if not self.use_altitude:
            return None
        sample = self.source.sample_at(video_time_s)
        if sample is None:
            return None
        for value in (sample.alt_agl_m, sample.alt_rel_m):
            if value is not None and math.isfinite(value) and value > 0.0:
                return float(value)
        return None


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
    if kind == "mavlink":
        source = MavlinkFlightData(
            url=str(get_cfg(config, "flight_data.mavlink_url", "udpin:0.0.0.0:14550")),
            time_offset_s=float(get_cfg(config, "flight_data.time_offset_s", 0.0)),
            max_gap_s=float(get_cfg(config, "flight_data.max_gap_s", 1.0)),
        )
        return FlightPrior(source, config)
    raise ValueError(f"unknown flight_data.source {kind!r}")
