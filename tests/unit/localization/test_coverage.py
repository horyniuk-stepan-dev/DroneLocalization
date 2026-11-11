"""Coverage model + coverage state (vision-only plan phase 1.1 / 1.2)."""

import math

import numpy as np
import pytest

from src.localization.coverage import CoverageModel, CoverageMonitor, CoverageParams, LocalPlane

LAT0, LON0 = 48.30, 25.90


def _square_ll(plane, x0, y0, x1, y1):
    xs = np.array([x0, x1, x1, x0], dtype=float)
    ys = np.array([y0, y0, y1, y1], dtype=float)
    lat, lon = plane.to_ll(xs, ys)
    return np.column_stack([lat, lon])


@pytest.fixture
def plane():
    return LocalPlane(LAT0, LON0)


@pytest.fixture
def model(plane):
    # 4 km x 2 km coverage made of 8 overlapping 600 m frames per row.
    polys = []
    for i in range(8):
        for j in range(4):
            x, y = -2000 + i * 500, -1000 + j * 500
            polys.append(_square_ll(plane, x, y, x + 600, y + 600))
    m = CoverageModel(polys, cell_m=10.0)
    # Re-centre test coordinates on the model's own plane.
    return m


def _ll(model, x, y):
    lat, lon = model.plane.to_ll(x, y)
    return float(lat), float(lon)


def _xy_of(model, plane, x, y):
    lat, lon = plane.to_ll(x, y)
    mx, my = model.plane.to_xy(lat, lon)
    return float(mx), float(my)


def test_inside_distance_and_outside_return(model, plane):
    # Centre of the coverage: ~1100 m to the north/south edges (covered y = -1000..1100).
    lat, lon = plane.to_ll(0.0, 0.0)
    q = model.query(float(lat), float(lon))
    assert q.inside
    assert 1000 < q.dist_to_edge_m < 1200

    # 500 m east of the east edge (x = 2100): outside, return bearing ~west.
    lat, lon = plane.to_ll(2600.0, 0.0)
    q = model.query(float(lat), float(lon))
    assert not q.inside
    assert q.dist_to_edge_m == pytest.approx(-500, abs=25)
    assert q.return_bearing_deg == pytest.approx(270, abs=5)
    assert (
        model.contains(q.return_lat, q.return_lon)
        or model.query(q.return_lat, q.return_lon).dist_to_edge_m > -2 * model.cell_m
    )


def test_far_outside_grid_still_points_back(model, plane):
    lat, lon = plane.to_ll(0.0, 20000.0)  # 19 km north, beyond the 2 km pad
    q = model.query(float(lat), float(lon))
    assert not q.inside
    assert q.return_dist_m == pytest.approx(18900, rel=0.02)
    assert q.return_bearing_deg == pytest.approx(180, abs=3)


def test_fraction_inside(model, plane):
    fp_in = _square_ll(plane, -300, -300, 300, 300)
    assert model.fraction_inside(fp_in) == pytest.approx(1.0)
    fp_half = _square_ll(plane, 1800, -300, 2400, 300)  # edge at x=2100
    assert model.fraction_inside(fp_half) == pytest.approx(0.5, abs=0.15)
    fp_out = _square_ll(plane, 5000, 0, 5600, 600)
    assert model.fraction_inside(fp_out) == 0.0


def test_grid_cell_is_coarsened_to_cell_cap(plane):
    polys = [_square_ll(plane, 0, 0, 50_000, 50_000)]
    m = CoverageModel(polys, cell_m=5.0, max_cells=1_000_000)
    assert m.w * m.h <= 1_100_000
    assert m.cell_m > 5.0
    lat, lon = plane.to_ll(25_000.0, 25_000.0)
    assert m.query(float(lat), float(lon)).dist_to_edge_m == pytest.approx(25_000, rel=0.03)


def test_outline_and_geojson(model):
    assert len(model.outline_ll) == 1
    gj = model.outline_geojson()
    assert gj["geometry"]["type"] == "MultiPolygon"
    ring = gj["geometry"]["coordinates"][0][0]
    assert ring[0] == ring[-1]


def _ok(model, plane, x, y, source="keyframe", since=0.0, fov=True):
    lat, lon = plane.to_ll(x, y)
    r = {
        "success": True,
        "lat": float(lat),
        "lon": float(lon),
        "fix_source": source,
        "since_fix_s": since,
        "matched_frame": 1,
    }
    if source != "keyframe":
        r["is_of"] = True
    if fov and source == "keyframe":
        r["fov_polygon"] = [tuple(p) for p in _square_ll(plane, x - 150, y - 150, x + 150, y + 150)]
    return r


def test_exit_and_return_sequence(model, plane):
    """Fly east at 20 m/s out of the map, dead-reckon, lose it, come back."""
    mon = CoverageMonitor(model, CoverageParams())
    states, alerts = [], []
    t = 0.0
    x = 0.0
    # inside, approaching the east edge (x = 2100)
    while x < 2100 - 200:
        r = mon.update(_ok(model, plane, x, 0.0), t)
        states.append(r["coverage_state"])
        alerts.append((t, r["coverage_alert"], r["coverage_state"]))
        t += 1.0
        x += 20.0
    assert states[0] in ("inside", "unknown") or states[1] == "inside"
    assert "near_edge" in states
    first_warn = next(tt for tt, a, s in alerts if a)
    exit_t = (2100 - 0) / 20.0
    assert exit_t - first_warn >= 10.0  # warned well before the exit
    # keyframes fail, dead reckoning carries the position past the border
    since = 0.0
    while x < 2600:
        since += 1.0
        r = mon.update(_ok(model, plane, x, 0.0, "dead_reckoning", since), t)
        t += 1.0
        x += 20.0
    assert r["coverage_state"] == "outside"
    assert r["return_bearing_deg"] == pytest.approx(270, abs=5)
    assert r["fix_confirmed"] is False
    # dead reckoning expires: failures only
    for _ in range(40):
        r = mon.update({"success": False, "error": "Not enough valid inliers (3 < 12)"}, t)
        t += 1.0
    assert r["coverage_state"] == "unknown"
    assert "last_known_lat" in r
    # back inside: the first keyframe fix is not confirmed, the second consistent one is
    r1 = mon.update(_ok(model, plane, 1500, 0.0), t)
    t += 1.0
    r2 = mon.update(_ok(model, plane, 1480, 0.0), t)
    assert r1["fix_confirmed"] is False
    assert r1["coverage_state"] == "unknown"
    assert r2["fix_confirmed"] is True
    assert r2["coverage_state"] in ("inside", "near_edge")


def test_control_route_no_alert(model, plane):
    mon = CoverageMonitor(model, CoverageParams())
    alerts = 0
    t = 0.0
    for k in range(200):
        x = 600 * math.cos(k / 20.0)
        y = 300 * math.sin(k / 20.0)
        r = mon.update(_ok(model, plane, x, y), t)
        alerts += r["coverage_alert"]
        t += 0.5
    # occasional failed keyframes deep inside must not alert either
    for _ in range(3):
        r = mon.update({"success": False, "error": "No candidates found"}, t)
        alerts += r["coverage_alert"]
        t += 0.5
    assert alerts == 0
    assert r["coverage_state"] == "inside"


def test_lost_inside_distinguished_from_outside(model, plane):
    mon = CoverageMonitor(model, CoverageParams())
    t = 0.0
    for _ in range(5):
        mon.update(_ok(model, plane, 0.0, 0.0), t)
        t += 1.0
    for _ in range(6):
        r = mon.update({"success": False, "error": "Not enough valid inliers"}, t)
        t += 1.0
    assert r["coverage_state"] == "lost_inside"
    assert r["coverage_alert"] is False


def test_hysteresis_no_flapping(model, plane):
    mon = CoverageMonitor(model, CoverageParams(warn_s=0.0))
    t = 0.0
    seq = []
    # hover around the 300 m warning line (edge x=2100 -> line x=1800), +-20 m
    for k in range(40):
        x = 1800 + (20 if k % 2 else -20)
        r = mon.update(_ok(model, plane, x, 0.0), t)
        seq.append(r["coverage_state"])
        t += 1.0
    changes = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    assert changes <= 1


def test_dead_reckoning_within_uncertainty_of_border_is_leaving(model, plane):
    mon = CoverageMonitor(model, CoverageParams())
    t = 0.0
    for x in (1700.0, 1800.0, 1900.0, 2000.0):
        mon.update(_ok(model, plane, x, 0.0), t)
        t += 1.0
    # 20 m past the east edge (x = 2100), 4 s of dead reckoning: uncertainty 25 m
    r = mon.update(_ok(model, plane, 2120.0, 0.0, "dead_reckoning", 4.0), t)
    assert r["coverage_state"] == "leaving"
    t += 1.0
    r = mon.update(_ok(model, plane, 2300.0, 0.0, "dead_reckoning", 5.0), t)
    assert r["coverage_state"] == "outside"


def test_optical_flow_inherits_unconfirmed_anchor(model, plane):
    mon = CoverageMonitor(model, CoverageParams())
    mon.update(_ok(model, plane, 0.0, 0.0), 0.0)
    for k in range(40):
        mon.update({"success": False, "error": "Not enough valid inliers"}, 1.0 + k)
    r1 = mon.update(_ok(model, plane, 0.0, 0.0), 50.0)
    r_of = mon.update(_ok(model, plane, 5.0, 0.0, "optical_flow"), 50.2)
    assert r1["fix_confirmed"] is False
    assert r_of["fix_confirmed"] is False
    r2 = mon.update(_ok(model, plane, 20.0, 0.0), 51.0)
    r_of2 = mon.update(_ok(model, plane, 25.0, 0.0, "optical_flow"), 51.2)
    assert r2["fix_confirmed"] is True and r_of2["fix_confirmed"] is True
