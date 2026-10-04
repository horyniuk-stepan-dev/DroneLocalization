"""Terrain terms on a physically simulated flight (constant altitude over a hill).

The scene is generated with a perspective nadir camera, not with the model's
own formulas: edges are homographies fitted to projected ground points of the
overlap, anchors are affines fitted to the centre and corner rays (like the
simulator's calibration). Without terrain terms the chain keeps the anchor
scale across the hill; with them the error between the two anchors drops.
"""

import math

import cv2
import numpy as np
import pytest

from src.geometry.coordinates import CoordinateConverter
from src.geometry.dem import RasterDEM, lonlat_to_mercator
from src.geometry.pose_graph_optimizer import PoseGraphOptimizer, homography_to_similarity
from src.geometry.terrain_scale import (
    AltitudeProfile,
    TerrainScaleModel,
    TerrainTerms,
    anchor_altitudes,
    fit_constant_altitude,
    frame_grid,
    states_to_affines,
)

LAT0, LON0 = 48.42, 26.18
OX, OY = (float(v) for v in lonlat_to_mercator(LON0, LAT0))
K = math.cos(math.radians(LAT0))
W, HH = 1280, 720
F = 1920.0
ALT = 1250.0
PC = np.array([W / 2.0, HH / 2.0])


def terrain(x, y):
    """Local projected coordinates -> height (m): a hill on the leg and a gentle slope."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return 250.0 + 160.0 * np.exp(-(((x - 1200.0) / 1600.0) ** 2)) - 0.01 * y


def make_dem(pixel=10.0, half_w=9000.0, half_h=5000.0):
    cols = int(2 * half_w / pixel)
    rows = int(2 * half_h / pixel)
    xs = -half_w + (np.arange(cols) + 0.5) * pixel
    ys = half_h - (np.arange(rows) + 0.5) * pixel
    gx, gy = np.meshgrid(xs, ys)
    return RasterDEM(terrain(gx, gy), (OX - half_w, OY + half_h), (pixel, pixel), "EPSG:3857")


def ground_of_pixel(C, pix):
    d = (np.atleast_2d(pix) - PC) / (F * K)
    X = C + d * (ALT - 250.0)
    for _ in range(30):
        X = C + d * (ALT - terrain(X[:, 0], X[:, 1]))[:, None]
    return X


def project(C, X):
    h = terrain(X[:, 0], X[:, 1])
    return PC + F * K * (X - C) / (ALT - h)[:, None]


def frame_affine(C):
    pix = np.array([[W / 2, HH / 2], [0, 0], [W, 0], [W, HH], [0, HH]], dtype=np.float64)
    X = ground_of_pixel(C, pix)
    A = np.column_stack([pix, np.ones(len(pix))])
    sol, *_ = np.linalg.lstsq(A, X, rcond=None)
    return sol.T  # 2x3, local projected metres


def edge_similarity(C_from, C_to):
    gx, gy = np.meshgrid(np.linspace(20, W - 20, 24), np.linspace(20, HH - 20, 14))
    pix_to = np.column_stack([gx.ravel(), gy.ravel()])
    X = ground_of_pixel(C_to, pix_to)
    pix_from = project(C_from, X)
    inside = (
        (pix_from[:, 0] >= 0)
        & (pix_from[:, 0] <= W)
        & (pix_from[:, 1] >= 0)
        & (pix_from[:, 1] <= HH)
    )
    Hm, _ = cv2.findHomography(pix_to[inside], pix_from[inside], 0)
    return homography_to_similarity(Hm, W, HH)


def build_graph(n=33, step=250.0):
    centres = [np.array([-4000.0 + k * step, 0.0]) for k in range(n)]
    opt = PoseGraphOptimizer(W, HH, isotropy_weight=10.0)
    for k in range(n):
        opt.add_node(k)
    for k in range(n - 1):
        opt.add_edge(k, k + 1, edge_similarity(centres[k], centres[k + 1]), weight=8.0)
    for k in range(n - 2):
        opt.add_edge(k, k + 2, edge_similarity(centres[k], centres[k + 2]), weight=4.0)
    for k in (0, n - 1):
        opt.fix_node(k, frame_affine(centres[k]))
    opt.initialize_from_bfs()
    return opt, centres


def centre_errors(opt, centres):
    states = opt.node_states()
    return np.array([np.hypot(*(states[k][:2] - centres[k])) * K for k in range(len(centres))])


def model_for(opt, dem=None, f_px=F):
    return TerrainScaleModel(
        dem=dem or make_dem(),
        converter=CoordinateConverter("WEB_MERCATOR"),
        origin_xy=(OX, OY),
        frame_w=W,
        frame_h=HH,
        f_px=f_px,
        sign=opt.sign,
    )


def run_terrain(opt, model, mode="piecewise", weight=10.0, edge_correction=True, passes=2):
    anchors = anchor_altitudes(model, opt.anchor_states())
    profile = AltitudeProfile(anchors["ids"], anchors["alt"], mode=mode)
    terms = TerrainTerms(model, profile, prior_weight=weight, edge_correction=edge_correction)
    stats = None
    for _ in range(passes):
        stats = terms.apply(opt)
        opt.optimize(use_analytic_jac=True)
    return anchors, stats


@pytest.fixture(scope="module")
def baseline():
    opt, centres = build_graph()
    opt.optimize(use_analytic_jac=True)
    return centres, centre_errors(opt, centres)


def test_scene_reproduces_the_relief_failure(baseline):
    _centres, err = baseline
    assert err[0] < 3.0 and err[-1] < 3.0  # anchor affine fitted to 5 rays over relief
    assert err.max() > 60.0  # chain keeps the anchor scale over a 160 m hill


def test_anchor_altitude_recovers_flight_height():
    opt, _ = build_graph()
    anchors = anchor_altitudes(model_for(opt), opt.anchor_states())
    assert np.all(np.abs(anchors["alt"] - ALT) < 5.0)


# Measured 2026-10-04 (max error, baseline 149.6 m): piecewise/constant + edges
# 3.5 / 2.8 m, edges only 8.8 m, prior only (w=50) 35.6 m.
@pytest.mark.parametrize(
    "mode,weight,edge_correction,max_frac",
    [
        ("piecewise", 10.0, True, 0.05),
        ("constant", 10.0, True, 0.05),
        ("piecewise", 0.0, True, 0.1),
        ("piecewise", 50.0, False, 0.3),
    ],
)
def test_terrain_terms_remove_most_of_the_error(baseline, mode, weight, edge_correction, max_frac):
    _centres, err0 = baseline
    opt, centres = build_graph()
    opt.optimize(use_analytic_jac=True)
    _anchors, stats = run_terrain(opt, model_for(opt), mode, weight, edge_correction)
    err = centre_errors(opt, centres)
    assert stats["nodes_with_dem"] == len(centres)
    assert err.max() < max_frac * err0.max(), (err.max(), err0.max())


def test_wrong_focal_still_helps(baseline):
    _centres, err0 = baseline
    opt, centres = build_graph()
    opt.optimize(use_analytic_jac=True)
    run_terrain(opt, model_for(opt, f_px=F * 1.25))
    assert centre_errors(opt, centres).max() < 0.25 * err0.max()  # measured 22.9 m


def test_flat_dem_changes_nothing_measurable():
    opt, centres = build_graph(n=9)
    opt.optimize(use_analytic_jac=True)
    before = centre_errors(opt, centres)
    flat = RasterDEM(np.full((100, 100), 300.0), (OX - 9000, OY + 5000), (180.0, 100.0))
    opt2, _ = build_graph(n=9)
    opt2.optimize(use_analytic_jac=True)
    _a, stats = run_terrain(opt2, model_for(opt2, dem=flat))
    # flat DEM: no relief correction on edges and the prior agrees with the anchors
    assert stats["edge_scale_fix_p95"] < 1e-9
    assert abs(centre_errors(opt2, centres).max() - before.max()) < 0.5 * before.max() + 5.0


def test_missing_dem_skips_nodes():
    opt, _ = build_graph(n=9)
    far = RasterDEM(np.full((10, 10), 300.0), (OX + 50_000, OY + 50_000), (10.0, 10.0))
    model = model_for(opt, dem=far)
    anchors = anchor_altitudes(model, opt.anchor_states())
    assert not np.any(np.isfinite(anchors["alt"]))


def test_altitude_profile_modes():
    p = AltitudeProfile([10, 20], [1000.0, 1100.0])
    np.testing.assert_allclose(p([0, 10, 15, 20, 30]), [1000, 1000, 1050, 1100, 1100])
    c = AltitudeProfile([10, 20, 30], [1000.0, 1100.0, 1010.0], mode="constant")
    np.testing.assert_allclose(c([0, 25]), [1010.0, 1010.0])
    with pytest.raises(ValueError):
        AltitudeProfile([1], [np.nan])


def test_fit_constant_altitude_diagnostic():
    d = np.array([200.0, 260.0, 330.0, 400.0])
    g = (1250.0 - d) / 1920.0
    f, h = fit_constant_altitude(g, d)
    assert abs(f - 1920.0) < 1e-6 and abs(h - 1250.0) < 1e-6
    assert fit_constant_altitude(g[:2], d[:2]) is None


def test_states_to_affines_matches_model_and_grid_weights():
    from src.geometry.pose_graph.model_5dof import _state_to_affine

    st = np.array([[10.0, -5.0, np.log(0.7), np.log(0.72), 0.4]])
    for sign in (1.0, -1.0):
        np.testing.assert_allclose(
            states_to_affines(st, W / 2, HH / 2, sign)[0],
            _state_to_affine(st[0], W / 2, HH / 2, sign),
        )
    pts, w = frame_grid(W, HH)
    assert pts.shape == (144, 2) and abs(w.sum() - 1.0) < 1e-12


class TestPipelineWiring:
    """PropagationPipeline._build_terrain_terms: project camera, DEM request, profile."""

    def _pipeline(self, tmp_path, converter=True, **cfg):
        pytest.importorskip("faiss")
        pytest.importorskip("h5py")
        import json
        import types

        from src.workers.propagation_pipeline import PropagationPipeline

        project = tmp_path / "proj"
        db_dir = project / "sources" / "main"
        db_dir.mkdir(parents=True)
        (project / "project.json").write_text(
            json.dumps({"focal_length_mm": 13.2, "sensor_width_mm": 8.8}), encoding="utf-8"
        )

        class DB:
            metadata = {"frame_width": W, "frame_height": HH}
            db_path = str(db_dir / "database.h5")

            def get_num_frames(self):
                return 33

        cal = types.SimpleNamespace(
            converter=CoordinateConverter("WEB_MERCATOR") if converter else None, anchors=[]
        )
        config = {"graph_optimization": {"terrain_scale_prior": True, **cfg}}
        p = PropagationPipeline(DB(), cal, None, config=config)
        p._origin_xy = (OX, OY)
        return p, project

    def test_builds_terms_and_fixes_the_leg(self, tmp_path, monkeypatch, baseline):
        import src.geometry.dem as dem_mod

        seen = {}

        def fake_resolve(source, path, zoom, bbox, cache, **kw):
            seen.update(source=source, zoom=zoom, bbox=bbox, cache=cache)
            return make_dem()

        monkeypatch.setattr(dem_mod, "resolve_dem", fake_resolve)
        p, project = self._pipeline(tmp_path)
        assert p.terrain_prior and p._terrain_focal_px() == pytest.approx(F)
        opt, centres = build_graph()
        opt.optimize(use_analytic_jac=True)
        terms = p._build_terrain_terms(opt)
        assert terms is not None
        assert seen["source"] == "terrarium" and seen["zoom"] == 13
        assert seen["cache"] == project / "terrain"
        lat_min, lon_min, lat_max, lon_max = seen["bbox"]
        assert lat_min < LAT0 < lat_max and lon_min < LON0 < lon_max
        for _ in range(p.terrain_iterations):
            terms.apply(opt)
            opt.optimize(use_analytic_jac=True)
        assert centre_errors(opt, centres).max() < 0.05 * baseline[1].max()

    def test_no_converter_or_no_dem_is_skipped(self, tmp_path, monkeypatch):
        import src.geometry.dem as dem_mod

        opt, _ = build_graph(n=5)
        p, _project = self._pipeline(tmp_path, converter=False)
        assert p._build_terrain_terms(opt) is None
        monkeypatch.setattr(dem_mod, "resolve_dem", lambda *a, **k: None)
        p2, _ = self._pipeline(tmp_path / "b")
        assert p2._build_terrain_terms(opt) is None

    def test_focal_from_config_wins(self, tmp_path):
        p, _ = self._pipeline(tmp_path, terrain_focal_px=1500.0)
        assert p._terrain_focal_px() == 1500.0


def test_terrain_following_is_detected():
    from src.geometry.terrain_scale import altitude_terrain_slope

    d = np.array([200.0, 260.0, 330.0, 400.0])
    assert abs(altitude_terrain_slope(np.full(4, 1250.0) + [0.5, -1, 1, 0], d)) < 0.05
    assert altitude_terrain_slope(d + 1000.0, d) == pytest.approx(1.0)
    assert altitude_terrain_slope(d[:2] + 1000.0, d[:2]) is None
    assert altitude_terrain_slope(np.full(4, 1250.0), np.array([200.0, 201, 202, 203])) is None


def test_pipeline_skips_terrain_following(tmp_path, monkeypatch):
    """Anchors whose altitude tracks the DEM: the terms would only add error."""
    pytest.importorskip("faiss")
    pytest.importorskip("h5py")
    import src.geometry.dem as dem_mod
    import src.geometry.terrain_scale as ts

    monkeypatch.setattr(dem_mod, "resolve_dem", lambda *a, **k: make_dem())
    real = ts.anchor_altitudes

    def following(model, states):
        out = real(model, states)
        out["ids"] = [0, 1, 2]
        out["dem"] = np.array([200.0, 300.0, 400.0])
        out["alt"] = out["dem"] + 1000.0
        out["gsd"] = np.full(3, 0.5)
        return out

    monkeypatch.setattr(ts, "anchor_altitudes", following)
    p, _ = TestPipelineWiring()._pipeline(tmp_path)
    opt, _ = build_graph(n=5)
    assert p._build_terrain_terms(opt) is None
