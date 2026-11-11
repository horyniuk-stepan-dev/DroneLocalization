"""Map coverage model and coverage state (vision-only plan, phase 1).

The localizer used to know only "N keyframes failed in a row". This module
gives it a geometric model of where the prepared map ends:

* ``CoverageModel`` — union of every georeferenced reference frame's ground
  footprint, rasterised on a local metric grid (true ground metres, independent
  of the layers' UTM / Web-Mercator projections), with distance-to-border
  fields inside and outside and, outside, the nearest covered cell (bearing and
  distance back to the map).
* ``CoverageMonitor`` — per reported position (or failure) the coverage state
  ``inside`` / ``near_edge`` / ``leaving`` / ``outside`` / ``lost_inside`` /
  ``unknown`` with hysteresis, time-to-edge and an alert flag on escalation.

Pure numpy + cv2: no torch / Qt, unit-testable anywhere.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

EARTH_R = 6_371_008.8

STATES = ("inside", "near_edge", "leaving", "outside", "lost_inside", "unknown")
# States that warn the operator (plan 1.4); lost_inside is reported but is the
# old "lost" situation, not a map-edge alert.
ALERT_STATES = ("near_edge", "leaving", "outside", "unknown")
# States in which the edge guards of plan 1.3 are active.
EDGE_STATES = ("near_edge", "leaving", "outside", "unknown")
_SEVERITY = {"inside": 0, "near_edge": 1, "leaving": 2, "outside": 3}
# States left only after reacquire_fixes mutually consistent keyframe fixes.
_REACQUIRE_FROM = ("outside", "lost_inside", "unknown")


# ── Local projection ────────────────────────────────────────────────────────


class LocalPlane:
    """Equirectangular tangent plane around (lat0, lon0) in ground metres.

    ``planar=True`` (LOCAL relative maps): lat/lon already are y/x map units.
    Over a 1000 km corridor the cos(lat) scale error is a few percent at the
    far ends (inference from the projection formula) — fine for an edge
    warning measured in hundreds of metres.
    """

    def __init__(self, lat0: float, lon0: float, planar: bool = False) -> None:
        self.lat0, self.lon0, self.planar = float(lat0), float(lon0), bool(planar)
        self._kx = EARTH_R * math.radians(1.0) * math.cos(math.radians(lat0))
        self._ky = EARTH_R * math.radians(1.0)

    def to_xy(self, lat, lon):
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        if self.planar:
            return lon, lat
        return (lon - self.lon0) * self._kx, (lat - self.lat0) * self._ky

    def to_ll(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.planar:
            return y, x
        return self.lat0 + y / self._ky, self.lon0 + x / self._kx


@dataclass
class CoverageQuery:
    inside: bool
    dist_to_edge_m: float  # signed: > 0 inside, < 0 outside (to the nearest covered point)
    return_bearing_deg: float | None  # outside: bearing to the nearest covered point
    return_dist_m: float | None  # outside: distance to it
    return_lat: float | None = None
    return_lon: float | None = None


# ── Coverage model ──────────────────────────────────────────────────────────


class CoverageModel:
    """Rasterised union of reference-frame footprints with distance fields."""

    def __init__(
        self,
        polygons_ll: list[np.ndarray],
        *,
        cell_m: float = 5.0,
        max_cells: int = 16_000_000,
        pad_m: float = 2000.0,
        planar: bool = False,
        layer_polygons_ll: dict[str, list[np.ndarray]] | None = None,
    ) -> None:
        polys = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polygons_ll]
        polys = [p for p in polys if len(p) >= 3 and np.isfinite(p).all()]
        if not polys:
            raise ValueError("No reference footprints to build the coverage from")
        allpts = np.vstack(polys)
        lat0 = float(np.mean(allpts[:, 0]))
        lon0 = float(np.mean(allpts[:, 1]))
        self.plane = LocalPlane(lat0, lon0, planar=planar)
        self.n_footprints = len(polys)

        xy = [np.column_stack(self.plane.to_xy(p[:, 0], p[:, 1])) for p in polys]
        stack = np.vstack(xy)
        xmin, ymin = stack.min(axis=0) - pad_m
        xmax, ymax = stack.max(axis=0) + pad_m
        cell = max(float(cell_m), 1e-6)
        area_cells = (xmax - xmin) * (ymax - ymin) / (cell * cell)
        if area_cells > max_cells:
            cell = math.sqrt((xmax - xmin) * (ymax - ymin) / max_cells)
            logger.info(
                f"Coverage grid: cell {cell_m:.1f} m would need {area_cells / 1e6:.1f} M "
                f"cells; using {cell:.1f} m"
            )
        self.cell_m = cell
        self.x0, self.y1 = float(xmin), float(ymax)
        self.w = int(math.ceil((xmax - xmin) / cell)) + 1
        self.h = int(math.ceil((ymax - ymin) / cell)) + 1

        self.mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self._fill(self.mask, xy)
        self.layer_outlines_ll: dict[str, list[np.ndarray]] = {}
        for sid, lp in (layer_polygons_ll or {}).items():
            lmask = np.zeros_like(self.mask)
            lxy = [
                np.column_stack(self.plane.to_xy(*np.asarray(p, dtype=np.float64).T[:2]))
                for p in lp
                if len(p) >= 3
            ]
            if lxy:
                self._fill(lmask, lxy)
                self.layer_outlines_ll[str(sid)] = self._outline(lmask)
        self.outline_ll = self._outline(self.mask)

        # Inside: distance (cells) to the nearest uncovered cell.
        self._d_in = cv2.distanceTransform(self.mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # Outside: distance to the nearest covered cell + which cell that is.
        inv = (1 - self.mask).astype(np.uint8)
        # (The distance itself is recomputed exactly from the nearest cell.)
        _d_out, labels = cv2.distanceTransformWithLabels(
            inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        del _d_out
        # Labels number the zero pixels of ``inv`` (= covered cells) in raster order.
        rows, cols = np.nonzero(inv == 0)
        self._label_rc = np.column_stack([rows, cols]).astype(np.int32)
        self._labels = labels
        self.area_km2 = float(self.mask.sum()) * cell * cell / 1e6

    # ── grid helpers ──
    def _fill(self, mask: np.ndarray, polys_xy) -> None:
        # One polygon per call: a single fillPoly over many contours uses the
        # even-odd rule and would punch holes where footprints overlap.
        for p in polys_xy:
            pts = self._to_px_int(p)
            if len(p) == 4:
                cv2.fillConvexPoly(mask, pts, 1)
            else:
                cv2.fillPoly(mask, [pts], 1)

    def _to_px(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        return np.column_stack(
            [(xy[:, 0] - self.x0) / self.cell_m, (self.y1 - xy[:, 1]) / self.cell_m]
        )

    def _to_px_int(self, xy: np.ndarray) -> np.ndarray:
        return np.round(self._to_px(xy)).astype(np.int32).reshape(-1, 1, 2)

    def _px_to_xy(self, col, row):
        return self.x0 + np.asarray(col) * self.cell_m, self.y1 - np.asarray(row) * self.cell_m

    def _outline(self, mask: np.ndarray) -> list[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rings = []
        for c in contours:
            c = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
            if len(c) < 3:
                continue
            x, y = self._px_to_xy(c[:, 0], c[:, 1])
            lat, lon = self.plane.to_ll(x, y)
            rings.append(np.column_stack([lat, lon]))
        return rings

    def _cell(self, x: float, y: float) -> tuple[int, int, bool]:
        col = int(round((x - self.x0) / self.cell_m))
        row = int(round((self.y1 - y) / self.cell_m))
        on_grid = 0 <= col < self.w and 0 <= row < self.h
        return min(max(row, 0), self.h - 1), min(max(col, 0), self.w - 1), on_grid

    # ── queries ──
    def contains(self, lat: float, lon: float) -> bool:
        x, y = self.plane.to_xy(lat, lon)
        row, col, on_grid = self._cell(float(x), float(y))
        return bool(on_grid and self.mask[row, col])

    def query(self, lat: float, lon: float) -> CoverageQuery:
        x, y = (float(v) for v in self.plane.to_xy(lat, lon))
        row, col, on_grid = self._cell(x, y)
        if on_grid and self.mask[row, col]:
            return CoverageQuery(True, float(self._d_in[row, col]) * self.cell_m, None, None)
        label = int(self._labels[row, col])
        if label <= 0 or label > len(self._label_rc):
            return CoverageQuery(False, -float("inf"), None, None)
        r, c = self._label_rc[label - 1]
        tx, ty = self._px_to_xy(c, r)
        dx, dy = float(tx) - x, float(ty) - y
        dist = math.hypot(dx, dy)
        bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
        rlat, rlon = self.plane.to_ll(tx, ty)
        return CoverageQuery(False, -dist, bearing, dist, float(rlat), float(rlon))

    def fraction_inside(self, polygon_ll, samples: int = 7) -> float | None:
        """Share of a (quadrilateral) ground footprint lying on covered cells."""
        poly = np.asarray(polygon_ll, dtype=np.float64).reshape(-1, 2)
        if len(poly) != 4 or not np.isfinite(poly).all():
            return None
        x, y = self.plane.to_xy(poly[:, 0], poly[:, 1])
        p = np.column_stack([x, y])
        u = (np.arange(samples) + 0.5) / samples
        uu, vv = np.meshgrid(u, u)
        uu, vv = uu.ravel()[:, None], vv.ravel()[:, None]
        pts = (
            (1 - uu) * (1 - vv) * p[0]
            + uu * (1 - vv) * p[1]
            + uu * vv * p[2]
            + (1 - uu) * vv * p[3]
        )
        px = self._to_px(pts)
        cols = np.round(px[:, 0]).astype(int)
        rows = np.round(px[:, 1]).astype(int)
        ok = (cols >= 0) & (cols < self.w) & (rows >= 0) & (rows < self.h)
        hit = np.zeros(len(pts), dtype=bool)
        hit[ok] = self.mask[rows[ok], cols[ok]] > 0
        return float(hit.mean())

    def outline_geojson(self) -> dict:
        def ring(r):
            coords = [[float(lon), float(lat)] for lat, lon in r]
            return coords + coords[:1]

        return {
            "type": "Feature",
            "properties": {"kind": "map_coverage", "cell_m": self.cell_m},
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [[ring(r)] for r in self.outline_ll],
            },
        }


# ── Building from databases ─────────────────────────────────────────────────


def frame_footprints_ll(database, calibration) -> list[np.ndarray]:
    """Ground corners (lat, lon) of every georeferenced reference frame of one layer."""
    from src.geometry.frame_transform import get_frame_transform, project_frame_points

    converter = getattr(calibration, "converter", None)
    if converter is None:
        return []
    n = int(database.get_num_frames()) if hasattr(database, "get_num_frames") else 0
    support = getattr(database, "is_frame_georef_supported", None)
    out: list[np.ndarray] = []
    for fid in range(n):
        T = get_frame_transform(database, fid)
        if T is None:
            continue
        if support is not None:
            try:
                if not support(fid):
                    continue
            except Exception:
                pass
        h, w = database.get_frame_size(fid)
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
        try:
            T = np.asarray(T, dtype=np.float64)
            if T.shape == (3, 3):
                metric = project_frame_points(T, corners, w, h)
            else:
                metric = corners @ T[:, :2].T + T[:, 2]
            to_ll = getattr(converter, "metric_to_gps_array", None)
            if callable(to_ll):
                lat, lon = to_ll(metric[:, 0], metric[:, 1])
            else:
                lat, lon = np.array(
                    [converter.metric_to_gps(float(x), float(y)) for x, y in metric]
                ).T
        except Exception:
            continue
        poly = np.column_stack([lat, lon])
        if np.isfinite(poly).all():
            out.append(poly)
    return out


def build_coverage(localizer, config) -> CoverageModel | None:
    """Coverage of every layer the localizer can search (or its single database)."""
    from config import get_cfg

    t0 = time.perf_counter()
    layers: dict[str, list[np.ndarray]] = {}
    planar = False
    dbm, cm = getattr(localizer, "db_manager", None), getattr(localizer, "calib_manager", None)
    if dbm is not None and cm is not None:
        for sid in dbm.all_source_ids:
            db = dbm.get_database(sid)
            try:
                cal = cm.get(sid)
            except Exception:
                cal = None
            if db is None or cal is None:
                continue
            layers[str(sid)] = frame_footprints_ll(db, cal)
    if not any(layers.values()):
        cal = getattr(localizer, "calibration", None)
        planar = getattr(getattr(cal, "converter", None), "mode", "") == "LOCAL"
        layers = {"main": frame_footprints_ll(localizer.database, cal)}
    polys = [p for lp in layers.values() for p in lp]
    if not polys:
        logger.warning("Coverage model: no georeferenced reference frames — coverage disabled")
        return None
    model = CoverageModel(
        polys,
        cell_m=float(get_cfg(config, "localization.coverage.cell_m", 5.0)),
        max_cells=int(get_cfg(config, "localization.coverage.max_cells", 16_000_000)),
        planar=planar,
        layer_polygons_ll=layers if len(layers) > 1 else None,
    )
    logger.info(
        f"Coverage model: {model.n_footprints} footprints, {len(layers)} layer(s), "
        f"{model.area_km2:.1f} km², grid {model.w}x{model.h} @ {model.cell_m:.1f} m, "
        f"{(time.perf_counter() - t0) * 1e3:.0f} ms"
    )
    return model


# ── Coverage state ──────────────────────────────────────────────────────────


@dataclass
class CoverageParams:
    warn_m: float = 300.0
    warn_s: float = 15.0
    hyst_m: float = 50.0
    hyst_s: float = 5.0
    leave_frac: float = 0.5
    leave_hyst: float = 0.1
    lost_after_s: float = 3.0
    unknown_after_s: float = 30.0
    velocity_window_s: float = 3.0
    clear_count: int = 2
    reacquire_fixes: int = 2
    reacquire_tol_m: float = 30.0
    dr_base_m: float = 20.0  # edge_along_border_v2: DR 25-33 m off after ~2 s
    dr_drift_m_per_s: float = 5.0  # measured on the edge routes 2026-11-10

    @classmethod
    def from_config(cls, config) -> CoverageParams:
        from config import get_cfg

        p = cls()
        for k in cls.__dataclass_fields__:
            p.__setattr__(
                k, type(getattr(p, k))(get_cfg(config, f"localization.coverage.{k}", getattr(p, k)))
            )
        return p


class CoverageMonitor:
    """Coverage state per localization result (plan 1.2) with hysteresis."""

    def __init__(self, model: CoverageModel, params: CoverageParams | None = None) -> None:
        self.model = model
        self.p = params or CoverageParams()
        self.reset()

    def reset(self) -> None:
        self.state = "unknown"
        self._started = False
        self._hist: deque = deque()  # (t, x, y, signed_dist)
        self._last_pos: tuple[float, float, float] | None = None  # (t, lat, lon)
        self._last_kf: tuple[float, float, float] | None = None  # (t, x, y) of last keyframe fix
        self._last_query: CoverageQuery | None = None
        self._footprint_rel: np.ndarray | None = None  # last KF footprint, metres from centre
        self._lower_streak = 0
        self._reacq = 0
        self._anchor_confirmed = True
        self._last_fix_t: float | None = None
        self._clock = 0.0

    @property
    def edge_active(self) -> bool:
        return self.state in EDGE_STATES

    # ── helpers ──
    def _velocity_terms(self, t: float) -> tuple[float | None, float | None]:
        """(closing speed toward the border m/s (>0 = approaching), ground speed)."""
        h = [e for e in self._hist if t - e[0] <= self.p.velocity_window_s]
        if len(h) < 3:
            return None, None
        a = np.asarray(h, dtype=np.float64)
        tt = a[:, 0] - a[:, 0].mean()
        den = float(tt @ tt)
        if den < 1e-6:
            return None, None
        vx = float(tt @ (a[:, 1] - a[:, 1].mean())) / den
        vy = float(tt @ (a[:, 2] - a[:, 2].mean())) / den
        dd = float(tt @ (a[:, 3] - a[:, 3].mean())) / den
        return -dd, math.hypot(vx, vy)

    def _raw_geometric(self, q: CoverageQuery, tte: float | None, frac: float | None) -> str:
        cur = self.state
        warn_m = self.p.warn_m + (self.p.hyst_m if cur in ("near_edge", "leaving") else 0.0)
        warn_s = self.p.warn_s + (self.p.hyst_s if cur in ("near_edge", "leaving") else 0.0)
        leave = self.p.leave_frac + (self.p.leave_hyst if cur == "leaving" else 0.0)
        if frac is not None and frac < leave:
            return "leaving"
        if not q.inside:
            return "leaving"
        if q.dist_to_edge_m < warn_m or (tte is not None and tte < warn_s):
            return "near_edge"
        return "inside"

    def _set(self, raw: str, *, immediate: bool = False) -> bool:
        """Apply hysteresis; returns True on escalation into an alert state."""
        prev = self.state
        if raw == prev:
            self._lower_streak = 0
            return False
        ps, rs = _SEVERITY.get(prev), _SEVERITY.get(raw)
        lowering = ps is not None and rs is not None and rs < ps
        if lowering and not immediate:
            self._lower_streak += 1
            if self._lower_streak < self.p.clear_count:
                return False
        self._lower_streak = 0
        self.state = raw
        if raw != prev:
            logger.info(f"Coverage state: {prev} -> {raw}")
        return raw in ALERT_STATES and (
            prev not in ALERT_STATES
            or prev == "unknown"
            or _SEVERITY.get(raw, 9) > _SEVERITY.get(prev, 9)
        )

    def _uncertainty(self, now: float, since_fix: float | None = None) -> float:
        """Radius of a position not confirmed by the map, from its age."""
        if since_fix is None or since_fix <= 0:
            since_fix = 0.0 if self._last_fix_t is None else max(now - self._last_fix_t, 0.0)
        return self.p.dr_base_m + self.p.dr_drift_m_per_s * since_fix

    def _now(self, t: float | None, dt: float | None) -> float:
        if t is not None and np.isfinite(t):
            self._clock = float(t)
        else:
            self._clock += float(dt or 0.0)
        return self._clock

    # ── main entry ──
    def update(self, result: dict, t: float | None = None, dt: float | None = None) -> dict:
        """Annotate ``result`` in place with the coverage fields; return it."""
        now = self._now(t, dt)
        ok = bool(result.get("success")) and "lat" in result and "lon" in result
        if result.get("fallback_mode") == "retrieval_only":
            ok = False
        alert = False
        q = None
        tte = None
        frac = None
        if ok:
            lat, lon = float(result["lat"]), float(result["lon"])
            q = self.model.query(lat, lon)
            x, y = (float(v) for v in self.model.plane.to_xy(lat, lon))
            self._hist.append(
                (now, x, y, q.dist_to_edge_m if np.isfinite(q.dist_to_edge_m) else -1e9)
            )
            while self._hist and now - self._hist[0][0] > max(self.p.velocity_window_s * 2, 1.0):
                self._hist.popleft()
            closing, _speed = self._velocity_terms(now)
            if q.inside and closing is not None and closing > 0.5:
                tte = q.dist_to_edge_m / closing
            fix_source = str(result.get("fix_source", "keyframe"))
            is_kf = fix_source == "keyframe" and not result.get("is_of")
            fov = result.get("fov_polygon")
            if is_kf and fov and len(fov) == 4:
                frac = self.model.fraction_inside(fov)
                fx, fy = self.model.plane.to_xy(
                    np.asarray(fov, dtype=np.float64)[:, 0], np.asarray(fov, dtype=np.float64)[:, 1]
                )
                self._footprint_rel = np.column_stack([fx - x, fy - y])
            elif self._footprint_rel is not None:
                fp = self._footprint_rel + [x, y]
                flat, flon = self.model.plane.to_ll(fp[:, 0], fp[:, 1])
                frac = self.model.fraction_inside(np.column_stack([flat, flon]))
            since_fix = result.get("since_fix_s")
            since_fix = float(since_fix) if since_fix is not None else 0.0
            if fix_source == "dead_reckoning":
                unc = self._uncertainty(now, since_fix)
                if not q.inside:
                    # Within its own uncertainty of the border a dead-reckoned
                    # position cannot tell "outside" (edge_along_border_v2:
                    # 16 of 22 "outside" rows were 5-25 m inside, DR error 25-33 m).
                    raw = "outside" if -q.dist_to_edge_m > unc else "leaving"
                elif since_fix >= self.p.lost_after_s and q.dist_to_edge_m > self.p.warn_m:
                    raw = "lost_inside"
                else:
                    raw = self._raw_geometric(q, tte, frac)
            else:
                raw = self._raw_geometric(q, tte, frac)

            confirmed = True
            if is_kf:
                if self.state in _REACQUIRE_FROM and self._started:
                    consistent = False
                    if self._last_kf is not None:
                        lt, lx, ly = self._last_kf
                        gap = max(now - lt, 0.0)
                        _c, speed = self._velocity_terms(now)
                        allowed = self.p.reacquire_tol_m + (speed or 0.0) * gap
                        consistent = math.hypot(x - lx, y - ly) <= allowed and gap < 10.0
                    self._reacq = self._reacq + 1 if (consistent or self._reacq == 0) else 1
                    if self._reacq < self.p.reacquire_fixes:
                        confirmed = False
                        self._anchor_confirmed = False
                    else:
                        self._reacq = 0
                        alert = self._set(raw, immediate=True)
                else:
                    alert = self._set(raw)
                self._last_kf = (now, x, y)
                self._last_fix_t = now
                self._anchor_confirmed = confirmed
            else:
                # Optical flow tracks from the last keyframe: it is only as
                # confirmed as that keyframe.
                confirmed = self._anchor_confirmed
                if self.state not in _REACQUIRE_FROM or raw in ("outside", "lost_inside"):
                    # OF / DR positions may escalate or move between the lost
                    # states but never clear a lost/outside state by themselves.
                    alert = self._set(raw)
            self._started = True
            self._last_pos = (now, lat, lon)
            self._last_query = q
            result["fix_confirmed"] = bool(confirmed) and fix_source != "dead_reckoning"
            if fix_source in ("dead_reckoning", "optical_flow"):
                result["position_uncertainty_m"] = round(self._uncertainty(now, since_fix), 1)
        else:
            # Failure: predict from the last reported position.
            if self._last_pos is None:
                raw = "unknown"
            else:
                lt, llat, llon = self._last_pos
                age = now - lt
                result["last_known_lat"], result["last_known_lon"] = llat, llon
                result["since_position_s"] = round(age, 2)
                if age > self.p.unknown_after_s:
                    raw = "unknown"
                else:
                    q = self.model.query(llat, llon)
                    if not q.inside:
                        # Same rule as for dead reckoning: just past the border
                        # within the position's own uncertainty stays "leaving".
                        unc = self._uncertainty(now)
                        raw = "outside" if -q.dist_to_edge_m > unc else "leaving"
                    elif q.dist_to_edge_m > self.p.warn_m:
                        # Deep inside: a short gap keeps the state, a long one is "lost".
                        raw = "lost_inside" if age >= self.p.lost_after_s else self.state
                    else:
                        raw = self.state if self.state not in ("inside", "unknown") else "near_edge"
            if raw in ("outside", "unknown", "lost_inside"):
                alert = self._set(raw, immediate=True)
                self._reacq = 0
            else:
                alert = self._set(raw)
            self._started = True
        result["coverage_state"] = self.state
        if q is not None:
            result["dist_to_edge_m"] = (
                round(q.dist_to_edge_m, 1) if np.isfinite(q.dist_to_edge_m) else None
            )
            result["return_bearing_deg"] = (
                round(q.return_bearing_deg, 1) if q.return_bearing_deg is not None else None
            )
            result["return_dist_m"] = (
                round(q.return_dist_m, 1) if q.return_dist_m is not None else None
            )
            if q.return_lat is not None:
                result["return_lat"], result["return_lon"] = q.return_lat, q.return_lon
        if tte is not None:
            result["time_to_edge_s"] = round(tte, 1)
        if frac is not None:
            result["footprint_inside_frac"] = round(frac, 2)
        result["coverage_alert"] = bool(alert)
        return result

    def nearest_return_point(self) -> tuple[float, float] | None:
        q = self._last_query
        if q is not None and q.return_lat is not None:
            return q.return_lat, q.return_lon
        return None
