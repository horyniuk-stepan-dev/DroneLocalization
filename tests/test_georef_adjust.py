import numpy as np
import pytest

from src.geometry.georef_adjust import frame_centers, move_node, refit, with_moved_node
from src.geometry.georef_fit import apply
from src.geometry.georef_graph import GRAPH, build_local_graph, fit_graph_georef

_R = 6378137.0


class _Converter:
    """Equirectangular stand-in for CoordinateConverter (metres per degree)."""

    def gps_to_metric(self, lat, lon):
        return np.radians(lon) * _R, np.radians(lat) * _R

    def metric_to_gps(self, x, y):
        return np.degrees(y / _R), np.degrees(x / _R)

    def ground_scale_factor(self, lat=None):
        return 1.0


def _sim(scale, deg, t):
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.hstack([scale * R, np.asarray(t, float)[:, None]])


def _layer(n=36):
    """A chain like a real layer: every 5th frame valid, ~1/4 footprint apart."""
    ids = np.arange(n) * 5
    t = np.linspace(0, 1, n)
    local = np.column_stack([t * 40, 6 * np.sin(t * 3)])
    return build_local_graph(ids, local, footprint=4.0), ids, local


def _gcps(conv, M, local, idx, ids):
    out = []
    for i in idx:
        mx, my = apply(M, local[[i]])[0]
        lat, lon = conv.metric_to_gps(mx, my)
        out.append(
            {"name": f"Авто {ids[i]}", "x": local[i, 0], "y": local[i, 1], "lat": lat, "lon": lon}
        )
    return out


M = _sim(12.0, 20, [3.0e6, 5.5e6])


def test_moved_frame_lands_on_target_and_other_points_hold():
    conv = _Converter()
    g, ids, local = _layer()
    gcps = _gcps(conv, M, local, [0, 8, 16, 24, 30], ids)
    before = apply(M, local)
    target = before[35] + [72.0, 0.0]  # 1.5 footprints off
    lat, lon = conv.metric_to_gps(*target)

    mv = move_node(gcps, g, int(ids[35]), lat, lon, conv, 1)

    assert mv.fit.kind == GRAPH
    np.testing.assert_allclose(mv.centers[35], target, atol=0.5)
    assert mv.residual_m < 0.5
    assert mv.others_max_m < 5.0
    # the correction is local: frames before the last untouched point stay put
    shift = np.linalg.norm(mv.centers - before, axis=1)
    assert shift[:24].max() < 5.0
    assert shift[35] == pytest.approx(72.0, abs=0.5)
    # ... and it grows monotonically towards the moved frame
    assert np.all(np.diff(shift[30:]) > 0)
    assert mv.gcps[-1]["manual"] and mv.gcps[-1]["frame_id"] == ids[35]
    assert len(gcps) == 5  # input list is not modified


def test_move_replaces_point_of_same_frame_and_previous_move():
    conv = _Converter()
    g, ids, local = _layer()
    gcps = _gcps(conv, M, local, [0, 12, 24, 35], ids)
    t1 = apply(M, local[[35]])[0] + [30.0, 10.0]
    mv1 = move_node(gcps, g, int(ids[35]), *conv.metric_to_gps(*t1), conv, 1)
    assert mv1.replaced == [f"Авто {ids[35]}"]
    assert len(mv1.gcps) == 4

    t2 = t1 + [-10.0, 5.0]
    mv2 = move_node(mv1.gcps, g, int(ids[35]), *conv.metric_to_gps(*t2), conv, 1)
    assert mv2.replaced == [f"Кадр {ids[35]} (вручну)"]
    assert len(mv2.gcps) == 4
    np.testing.assert_allclose(mv2.centers[35], t2, atol=0.5)


def test_weight_beats_a_stiff_previous_fit():
    """A stored stiffness of 3 must not leave the moved frame tens of metres off."""
    conv = _Converter()
    g, ids, local = _layer()
    gcps = _gcps(conv, M, local, [0, 8, 16, 24, 30], ids)
    target = apply(M, local[[35]])[0] + [72.0, 0.0]
    mv = move_node(gcps, g, int(ids[35]), *conv.metric_to_gps(*target), conv, 1, stiffness=3.0)
    assert mv.fit.stiffness == pytest.approx(0.1)
    assert mv.residual_m < 0.5 and mv.others_max_m < 5.0


def test_two_points_fall_back_to_exact_similarity():
    conv = _Converter()
    g, ids, local = _layer()
    gcps = _gcps(conv, M, local, [0, 35], ids)
    target = apply(M, local[[35]])[0] + [20.0, -15.0]
    mv = move_node(gcps, g, int(ids[35]), *conv.metric_to_gps(*target), conv, 1)
    assert mv.fit.kind != GRAPH and len(mv.gcps) == 2
    np.testing.assert_allclose(mv.centers[35], target, atol=1e-3)
    np.testing.assert_allclose(mv.centers[0], apply(M, local[[0]])[0], atol=1e-3)


def test_refit_of_unmoved_points_reproduces_the_map():
    conv = _Converter()
    g, ids, local = _layer()
    gcps = _gcps(conv, M, local, [0, 8, 16, 24, 35], ids)
    fit, k = refit(gcps, g, conv, 1)
    np.testing.assert_allclose(frame_centers(fit, g), apply(M, local), atol=1e-2)
    assert k == 1.0 and all(r["residual_m"] < 1e-2 for r in gcps)


def test_unknown_frame_and_bad_coordinates():
    conv = _Converter()
    g, ids, local = _layer()
    gcps = _gcps(conv, M, local, [0, 16, 35], ids)
    with pytest.raises(ValueError):
        with_moved_node(gcps, g, 3, 48.0, 25.0)  # frame 3 is not a node
    with pytest.raises(ValueError):
        with_moved_node(gcps, g, int(ids[5]), float("nan"), 25.0)


def test_weights_default_is_unchanged_and_validated():
    g, ids, local = _layer()
    idx = [0, 8, 16, 24, 35]
    dst = apply(M, local[idx]) + np.random.default_rng(0).normal(0, 3, (5, 2))
    a = fit_graph_georef(g, local[idx], dst, stiffness=0.3)
    b = fit_graph_georef(g, local[idx], dst, stiffness=0.3, weights=np.ones(5))
    np.testing.assert_allclose(a.frame_matrices, b.frame_matrices)
    heavy = fit_graph_georef(g, local[idx], dst, stiffness=0.3, weights=[1, 1, 30, 1, 1])
    assert heavy.residuals[2] < 0.2 * a.residuals[2]
    with pytest.raises(ValueError):
        fit_graph_georef(g, local[idx], dst, weights=[1, 1, 0, 1, 1])
