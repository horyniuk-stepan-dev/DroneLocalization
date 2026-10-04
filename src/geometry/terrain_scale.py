"""Terrain (relief) terms for the propagation pose graph.

Problem
-------
Every keyframe is a 5-DoF affine (centre, two scales, angle): a flat ground
model. Over relief that model is biased in a way the edges cannot see. With
the drone at a constant altitude H (barometric hold, the usual survey mode and
what the FlightSimulator flies), a nadir camera sees ground at height h with
GSD (H - h) / f. Two overlapping frames look at the SAME ground, so the
homography between them has scale ratio 1, while the frames' own GSDs differ
by the terrain under each footprint. A temporal chain therefore keeps the
anchor's scale along the whole leg, and the position error grows with the
integral of the GSD error. On testtopboch_hh (AGL 746-992 m along the main
legs) this made the error between turn anchors grow to 460 m.

Terms (flag graph_optimization.terrain_scale_prior, default off)
---------------------------------------------------------------
* Altitude profile H(frame): H_a = GSD_a * f + D_a at every anchor (GSD_a from
  the anchor affine in ground metres, D_a = DEM under its footprint), linear
  between anchors in frame id (or one robust constant). A later telemetry
  stream can supply H(frame) directly through the same interface.
* Node prior: log GSD_k -> log((H(k) - D_k) / f), a unary factor per free node.
* Edge correction: an edge i->j measured over the overlap (mean height h_ov)
  is rewritten to what the per-frame model expects:
  scale  += log((H_j - D_j)/(H_i - D_i)) - log((H_j - h_ov)/(H_i - h_ov)),
  shift  *= (H_i - h_ov) / (H_i - D_i).
  With exact anchors at both ends, the corrected chain alone follows the relief.

D_k is the DEM under frame k weighted by r^2 from the frame centre, the
weighting a least-squares affine fit gives its points, so D_k matches what an
anchor affine fitted to corner points encodes. A wrong focal length scales only
the relief correction (by f_true/f_used), never the anchors' own scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.geometry.dem import RasterDEM, mercator_to_lonlat
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

PIECEWISE = "piecewise"
CONSTANT = "constant"
_MIN_VALID_FRACTION = 0.5
_MIN_OVERLAP_POINTS = 4
_MAX_LOG_SCALE_FIX = float(np.log(1.5))


def states_to_affines(states: np.ndarray, cx: float, cy: float, sign: float = 1.0) -> np.ndarray:
    """(N, 5) node states -> (N, 2, 3) affines (same formula as model_5dof)."""
    s = np.atleast_2d(np.asarray(states, dtype=np.float64))
    sx = np.exp(np.clip(s[:, 2], -30, 30))
    sy = np.exp(np.clip(s[:, 3], -30, 30))
    c, sn = np.cos(s[:, 4]), np.sin(s[:, 4])
    m = np.empty((len(s), 2, 3), dtype=np.float64)
    m[:, 0, 0] = c * sx
    m[:, 0, 1] = -sn * sign * sy
    m[:, 1, 0] = sn * sx
    m[:, 1, 1] = c * sign * sy
    m[:, 0, 2] = s[:, 0] - (m[:, 0, 0] * cx + m[:, 0, 1] * cy)
    m[:, 1, 2] = s[:, 1] - (m[:, 1, 0] * cx + m[:, 1, 1] * cy)
    return m


def frame_grid(width: int, height: int, nx: int = 16, ny: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre pixel grid (G, 2) over the frame and r^2 weights (G,), sum 1."""
    xs = (np.arange(nx) + 0.5) * width / nx
    ys = (np.arange(ny) + 0.5) * height / ny
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    r2 = (pts[:, 0] - width / 2.0) ** 2 + (pts[:, 1] - height / 2.0) ** 2
    return pts, r2 / r2.sum()


def metric_to_latlon(x, y, converter) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised converter.metric_to_gps (closed form for WEB_MERCATOR)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if str(getattr(converter, "mode", "WEB_MERCATOR")).upper() == "WEB_MERCATOR":
        lon, lat = mercator_to_lonlat(x, y)
        return lat, lon
    lat, lon = converter.metric_to_gps_array(x, y)
    return np.asarray(lat), np.asarray(lon)


def ground_scale(lat, converter) -> np.ndarray:
    """Projected metre -> ground metre: cos(lat) for WEB_MERCATOR, else 1."""
    lat = np.asarray(lat, dtype=np.float64)
    if str(getattr(converter, "mode", "WEB_MERCATOR")).upper() == "WEB_MERCATOR":
        return np.cos(np.radians(lat))
    return np.ones_like(lat)


class AltitudeProfile:
    """Flight altitude (DEM datum) per frame id.

    ``piecewise``: linear in frame id between anchors, constant outside.
    ``constant``: the median of all anchor estimates (pure barometric hold).
    """

    def __init__(self, frame_ids, altitudes, mode: str = PIECEWISE) -> None:
        ids = np.asarray(frame_ids, dtype=np.float64)
        alts = np.asarray(altitudes, dtype=np.float64)
        ok = np.isfinite(ids) & np.isfinite(alts)
        if not np.any(ok):
            raise ValueError("no anchor altitude available")
        order = np.argsort(ids[ok])
        self.frame_ids = ids[ok][order]
        self.altitudes = alts[ok][order]
        if mode not in (PIECEWISE, CONSTANT):
            raise ValueError(f"altitude mode must be {PIECEWISE!r} or {CONSTANT!r}")
        self.mode = mode
        self._const = float(np.median(self.altitudes))

    def __call__(self, frame_ids) -> np.ndarray:
        f = np.asarray(frame_ids, dtype=np.float64)
        if self.mode == CONSTANT or len(self.frame_ids) == 1:
            return np.full(f.shape, self._const if self.mode == CONSTANT else self.altitudes[0])
        return np.interp(f, self.frame_ids, self.altitudes)


def fit_constant_altitude(gsd_ground, dem_heights, min_spread_m: float = 30.0):
    """Diagnostic: GSD_a = (H - D_a)/f by least squares over anchors.

    Returns (f_px, H) or None when the anchors do not span enough relief.
    """
    g = np.asarray(gsd_ground, dtype=np.float64)
    d = np.asarray(dem_heights, dtype=np.float64)
    ok = np.isfinite(g) & np.isfinite(d)
    if ok.sum() < 3 or np.ptp(d[ok]) < min_spread_m:
        return None
    beta, alpha = np.polyfit(d[ok], g[ok], 1)
    if beta >= 0:
        return None
    return float(-1.0 / beta), float(-alpha / beta)


def altitude_terrain_slope(altitudes, dem_heights, min_spread_m: float = 30.0) -> float | None:
    """dH/dDEM over the anchors: ~0 for a barometric hold, ~1 for terrain following.

    The terrain terms assume H does not follow the ground between anchors; a
    slope near 1 means the flight kept a constant height above ground, where
    the flat per-frame model is already right and the terms would add error.
    None when there are fewer than 3 anchors or they span too little relief.
    """
    a = np.asarray(altitudes, dtype=np.float64)
    d = np.asarray(dem_heights, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(d)
    if ok.sum() < 3 or np.ptp(d[ok]) < min_spread_m:
        return None
    return float(np.polyfit(d[ok], a[ok], 1)[0])


@dataclass
class TerrainScaleModel:
    dem: RasterDEM
    converter: object
    origin_xy: tuple[float, float]
    frame_w: int
    frame_h: int
    f_px: float
    sign: float = 1.0
    grid: tuple[int, int] = (16, 9)
    _pts: np.ndarray = field(init=False, repr=False)
    _w: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not (self.f_px > 0):
            raise ValueError("focal length in pixels must be positive")
        self._pts, self._w = frame_grid(self.frame_w, self.frame_h, *self.grid)

    @property
    def cx(self) -> float:
        return self.frame_w / 2.0

    @property
    def cy(self) -> float:
        return self.frame_h / 2.0

    def sample_footprints(self, states: np.ndarray) -> dict:
        """DEM under each frame: heights (N, G), weighted mean D (N,), centre lat (N,)."""
        s = np.atleast_2d(np.asarray(states, dtype=np.float64))
        aff = states_to_affines(s, self.cx, self.cy, self.sign)
        pts_h = np.column_stack([self._pts, np.ones(len(self._pts))])  # (G, 3)
        ground = np.einsum("nij,gj->ngi", aff, pts_h)  # (N, G, 2) local metric
        gx = ground[..., 0] + self.origin_xy[0]
        gy = ground[..., 1] + self.origin_xy[1]
        lat, lon = metric_to_latlon(gx, gy, self.converter)
        heights = self.dem.sample(lat, lon)  # (N, G)
        valid = np.isfinite(heights)
        w = np.where(valid, self._w[None, :], 0.0)
        wsum = w.sum(axis=1)
        mean = np.where(
            (valid.mean(axis=1) >= _MIN_VALID_FRACTION) & (wsum > 0),
            np.nansum(np.where(valid, heights, 0.0) * w, axis=1) / np.maximum(wsum, 1e-12),
            np.nan,
        )
        clat, _clon = metric_to_latlon(
            s[:, 0] + self.origin_xy[0], s[:, 1] + self.origin_xy[1], self.converter
        )
        return {"heights": heights, "mean": mean, "lat": np.asarray(clat), "affines": aff}

    def ground_gsd(self, states: np.ndarray, lat) -> np.ndarray:
        """Geometric-mean GSD of the node affines in ground metres per pixel."""
        s = np.atleast_2d(np.asarray(states, dtype=np.float64))
        return np.exp(0.5 * (s[:, 2] + s[:, 3])) * ground_scale(lat, self.converter)

    def overlap_plane(
        self, fp: dict, idx_i: np.ndarray, idx_j: np.ndarray, dt_pix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Plane fit to the DEM over the overlap of frames i and j, in frame-i pixels.

        A homography between two views of non-planar ground is (close to) the
        homography of the best plane through the overlap. Returns, per edge, the
        plane height at frame j's centre (pixel ``centre + dt`` of frame i) and
        the plane gradient (m per frame-i pixel, shape (E, 2)). NaN when the
        overlap holds too few DEM samples; zero gradient when it is degenerate.
        """
        aff = fp["affines"]
        heights = fp["heights"]
        n = len(idx_i)
        h_at = np.full(n, np.nan)
        grad = np.zeros((n, 2))
        if n == 0:
            return h_at, grad
        pts_h = np.column_stack([self._pts, np.ones(len(self._pts))])
        ground_i = np.einsum("eij,gj->egi", aff[idx_i], pts_h)  # (E, G, 2)
        a_j = aff[idx_j]
        lin = a_j[:, :, :2]
        det = lin[:, 0, 0] * lin[:, 1, 1] - lin[:, 0, 1] * lin[:, 1, 0]
        ok_det = np.abs(det) > 1e-12
        inv = np.zeros_like(lin)
        inv[:, 0, 0] = lin[:, 1, 1]
        inv[:, 0, 1] = -lin[:, 0, 1]
        inv[:, 1, 0] = -lin[:, 1, 0]
        inv[:, 1, 1] = lin[:, 0, 0]
        inv /= np.where(ok_det, det, 1.0)[:, None, None]
        pix_j = np.einsum("eij,egj->egi", inv, ground_i - a_j[:, None, :, 2])
        inside = (
            (pix_j[..., 0] >= 0)
            & (pix_j[..., 0] <= self.frame_w)
            & (pix_j[..., 1] >= 0)
            & (pix_j[..., 1] <= self.frame_h)
            & ok_det[:, None]
        )
        h = heights[idx_i]
        good = inside & np.isfinite(h)
        cnt = good.sum(axis=1)
        enough = cnt >= _MIN_OVERLAP_POINTS
        wgt = good.astype(np.float64)
        hz = np.where(good, h, 0.0)
        # plane h = a + b*(u - cx) + c*(v - cy) by weighted least squares
        u = (self._pts[:, 0] - self.cx) / self.frame_w
        v = (self._pts[:, 1] - self.cy) / self.frame_w
        basis = np.stack([np.ones_like(u), u, v], axis=1)  # (G, 3)
        ata = np.einsum("eg,gk,gl->ekl", wgt, basis, basis)
        atb = np.einsum("eg,gk,eg->ek", wgt, basis, hz)
        mean = np.where(enough, atb[:, 0] / np.maximum(cnt, 1), np.nan)
        cond = np.linalg.cond(ata + np.eye(3)[None] * 1e-12)
        plane_ok = enough & (cnt >= 6) & (cond < 1e6)
        coef = np.zeros((n, 3))
        if np.any(plane_ok):
            coef[plane_ok] = np.linalg.solve(ata[plane_ok], atb[plane_ok][..., None])[..., 0]
        du = dt_pix[:, 0] / self.frame_w
        dv = dt_pix[:, 1] / self.frame_w
        h_at = np.where(plane_ok, coef[:, 0] + coef[:, 1] * du + coef[:, 2] * dv, mean)
        grad[plane_ok, 0] = coef[plane_ok, 1] / self.frame_w
        grad[plane_ok, 1] = coef[plane_ok, 2] / self.frame_w
        return h_at, grad


class TerrainTerms:
    """Applies the node prior and the edge correction to a PoseGraphOptimizer.

    ``apply`` may be called repeatedly (fixed point): the raw edge measurements
    are kept on the first call and every call recomputes the terms from the
    optimizer's current node states.
    """

    def __init__(
        self,
        model: TerrainScaleModel,
        profile: AltitudeProfile,
        prior_weight: float,
        edge_correction: bool = True,
        min_agl_m: float = 10.0,
    ) -> None:
        self.model = model
        self.profile = profile
        self.prior_weight = float(prior_weight)
        self.edge_correction = bool(edge_correction)
        self.min_agl_m = float(min_agl_m)
        self._raw: dict[int, tuple[float, float, float, float]] = {}

    def apply(self, optimizer) -> dict:
        states = optimizer.node_states()
        anchor_ids = set(optimizer.anchor_states())
        fids = sorted(states)
        stats = {"nodes": len(fids), "priors": 0, "edges_corrected": 0, "edges": 0}
        if not fids:
            optimizer.set_scale_priors({})
            return stats
        idx = {fid: k for k, fid in enumerate(fids)}
        mat = np.array([states[f] for f in fids], dtype=np.float64)
        fp = self.model.sample_footprints(mat)
        dem_mean = fp["mean"]
        alt = self.profile(np.array(fids, dtype=np.float64))
        agl = alt - dem_mean
        ok = np.isfinite(agl) & (agl > self.min_agl_m)
        stats["nodes_with_dem"] = int(np.isfinite(dem_mean).sum())

        priors: dict[int, tuple[float, float]] = {}
        if self.prior_weight > 0:
            k = ground_scale(fp["lat"], self.model.converter)
            target = np.log(np.where(ok, agl, 1.0) / (self.model.f_px * k))
            for fid in fids:
                n = idx[fid]
                if ok[n] and fid not in anchor_ids and optimizer.is_free(fid):
                    priors[fid] = (float(target[n]), self.prior_weight)
        optimizer.set_scale_priors(priors)
        stats["priors"] = len(priors)

        edges = [e for e in optimizer.edges if e.from_id in idx and e.to_id in idx]
        stats["edges"] = len(edges)
        if self.edge_correction and edges:
            for e in edges:
                self._raw.setdefault(id(e), (e.dtx, e.dty, e.log_dsx, e.log_dsy, e.dtheta))
            raw = np.array([self._raw[id(e)] for e in edges], dtype=np.float64)
            ii = np.array([idx[e.from_id] for e in edges])
            jj = np.array([idx[e.to_id] for e in edges])
            fixes = edge_corrections(
                raw,
                alt[ii],
                alt[jj],
                dem_mean[ii],
                dem_mean[jj],
                *self.model.overlap_plane(fp, ii, jj, raw[:, :2]),
                min_agl_m=self.min_agl_m,
            )
            valid = fixes["valid"] & ok[ii] & ok[jj]
            for n, e in enumerate(edges):
                r = raw[n]
                if valid[n]:
                    e.dtx, e.dty = fixes["dtx"][n], fixes["dty"][n]
                    e.log_dsx, e.log_dsy = fixes["log_dsx"][n], fixes["log_dsy"][n]
                    e.dtheta = fixes["dtheta"][n]
                else:
                    e.dtx, e.dty, e.log_dsx, e.log_dsy, e.dtheta = (float(v) for v in r)
            stats["edges_corrected"] = int(valid.sum())
            if valid.any():
                dls = 0.5 * (fixes["log_dsx"] + fixes["log_dsy"] - raw[:, 2] - raw[:, 3])
                stats["edge_scale_fix_p95"] = float(np.percentile(np.abs(dls[valid]), 95))
                stats["edge_shift_fix_p95"] = float(
                    np.percentile(np.abs(fixes["shift_factor"][valid] - 1.0), 95)
                )
                stats["edge_angle_fix_p95_deg"] = float(
                    np.degrees(np.percentile(np.abs(fixes["dtheta"] - raw[:, 4])[valid], 95))
                )
        if np.any(ok):
            stats["agl_min_m"] = float(np.min(agl[ok]))
            stats["agl_max_m"] = float(np.max(agl[ok]))
        return stats


def edge_corrections(
    raw: np.ndarray,
    alt_i: np.ndarray,
    alt_j: np.ndarray,
    dem_i: np.ndarray,
    dem_j: np.ndarray,
    h_at_j: np.ndarray,
    grad_i: np.ndarray,
    min_agl_m: float = 10.0,
) -> dict:
    """Rewrite raw edges (dtx, dty, log_dsx, log_dsy, dtheta) for the per-frame model.

    Measured (frame-i pixels): j's centre lies at dt = f(C_j - C_i)/(H_i - h~),
    h~ the overlap plane at C_j, and the linear part is
    rho * (I + dt g^T / (H_i - h~)) * R(dtheta), rho = (H_j - h~)/(H_i - h~),
    g the plane gradient per frame-i pixel. The per-frame model wants the shift
    in units of node i's GSD, (H_i - D_i)/f, and the scale ratio
    (H_j - D_j)/(H_i - D_i) without the slope term. Corrections are first order
    in the slope term and clamped.
    """
    raw = np.asarray(raw, dtype=np.float64)
    dt = raw[:, :2]
    with np.errstate(invalid="ignore", divide="ignore"):
        agl_i_ov = alt_i - h_at_j
        agl_j_ov = alt_j - h_at_j
        valid = (
            np.isfinite(h_at_j)
            & np.isfinite(dem_i)
            & np.isfinite(dem_j)
            & (agl_i_ov > min_agl_m)
            & (agl_j_ov > min_agl_m)
            & (alt_i - dem_i > min_agl_m)
            & (alt_j - dem_j > min_agl_m)
        )
        shift = np.where(valid, agl_i_ov / (alt_i - dem_i), 1.0)
        iso = np.where(
            valid,
            np.log((alt_j - dem_j) / (alt_i - dem_i)) - np.log(agl_j_ov / agl_i_ov),
            0.0,
        )
        denom = np.where(valid, agl_i_ov, 1.0)
    shift = np.clip(shift, 1.0 / 1.5, 1.5)
    iso = np.clip(iso, -_MAX_LOG_SCALE_FIX, _MAX_LOG_SCALE_FIX)
    g = np.where(valid[:, None], np.nan_to_num(grad_i), 0.0)
    # slope term E = dt g^T / (H_i - h~), expressed after R(dtheta): E' = R^T E R
    e = dt[:, :, None] * g[:, None, :] / denom[:, None, None]
    c, s_ = np.cos(raw[:, 4]), np.sin(raw[:, 4])
    rot = np.stack([np.stack([c, -s_], -1), np.stack([s_, c], -1)], -2)
    e_rot = np.einsum("eki,ekl,elj->eij", rot, e, rot)
    e_rot = np.clip(e_rot, -0.2, 0.2)
    return {
        "valid": valid,
        "dtx": dt[:, 0] * shift,
        "dty": dt[:, 1] * shift,
        "log_dsx": raw[:, 2] + iso - e_rot[:, 0, 0],
        "log_dsy": raw[:, 3] + iso - e_rot[:, 1, 1],
        "dtheta": raw[:, 4] - e_rot[:, 1, 0],
        "shift_factor": shift,
    }


def anchor_altitudes(model: TerrainScaleModel, anchor_states: dict[int, np.ndarray]) -> dict:
    """H_a = ground GSD_a * f + D_a for every anchor with DEM under it."""
    fids = sorted(anchor_states)
    if not fids:
        return {"ids": [], "alt": np.zeros(0), "gsd": np.zeros(0), "dem": np.zeros(0)}
    mat = np.array([anchor_states[f] for f in fids], dtype=np.float64)
    fp = model.sample_footprints(mat)
    gsd = model.ground_gsd(mat, fp["lat"])
    alt = gsd * model.f_px + fp["mean"]
    return {"ids": fids, "alt": alt, "gsd": gsd, "dem": fp["mean"]}
