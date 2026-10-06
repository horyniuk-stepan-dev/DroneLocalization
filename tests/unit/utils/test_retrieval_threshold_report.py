"""scripts/retrieval_threshold_report.py: footprint test, scoring, threshold suggestions."""

import json

import h5py
import numpy as np
import pytest

from scripts import retrieval_threshold_report as R
from src.geometry.dem import lonlat_to_mercator, mercator_to_lonlat

W, H = 1280, 720
LAT, LON = 48.42, 26.18


def _db(path, n=20, gsd=1.0, spacing=2000.0):
    """n frames along x, 2 km apart (no overlap); frame k is centred at x0 + k*spacing."""
    x0, y0 = (float(v) for v in lonlat_to_mercator(LON, LAT))
    aff = np.zeros((n, 2, 3))
    for k in range(n):
        cx, cy = x0 + k * spacing, y0
        aff[k] = [[gsd, 0.0, cx - gsd * W / 2], [0.0, -gsd, cy + gsd * H / 2]]
    status = np.ones(n, dtype=np.uint8)
    status[-1] = 3  # last frame not georeferenced
    with h5py.File(path, "w") as f:
        m = f.create_group("metadata")
        m.attrs["frame_width"], m.attrs["frame_height"] = W, H
        g = f.create_group("calibration")
        g.create_dataset("frame_affine", data=aff)
        g.create_dataset("frame_georef_status", data=status)
        g.attrs["projection_json"] = json.dumps({"mode": "WEB_MERCATOR", "reference_gps": None})
    return x0, y0


def _gps(x, y):
    lon, lat = mercator_to_lonlat(x, y)
    return float(lat), float(lon)


def test_footprint_contains(tmp_path):
    db = tmp_path / "db.h5"
    x0, y0 = _db(db)
    fp = R.load_footprints(db)
    assert fp.contains(0, *_gps(x0, y0)) is True
    assert fp.contains(0, *_gps(x0 + 600, y0 - 300)) is True
    assert fp.contains(0, *_gps(x0 + 700, y0)) is False
    assert fp.contains(0, *_gps(x0 + 600, y0), margin=0.1) is False
    assert fp.contains(19, *_gps(x0, y0)) is None  # not georeferenced
    assert fp.contains(99, *_gps(x0, y0)) is None


def _report(db, x0, y0, n_rows=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        k = i % 18
        lat, lon = _gps(x0 + k * 2000.0 + rng.uniform(-300, 300), y0 + rng.uniform(-150, 150))
        good = [k, round(float(rng.uniform(0.85, 0.97)), 4)]
        bad = [(k + 5) % 18, round(float(rng.uniform(0.55, 0.84)), 4)]
        # every 4th frame retrieves the wrong frame first
        first = [bad, good] if i % 4 == 0 else [good, bad]
        if i % 4 == 0:
            first[0][1] = round(float(rng.uniform(0.60, 0.80)), 4)
        rows.append(
            {
                "gt_valid": True,
                "gt_lat": lat,
                "gt_lon": lon,
                "retrieval_calls": [{"main": first}, {"main": [good]}],
            }
        )
    rows.append(
        {
            "gt_valid": False,
            "gt_lat": None,
            "gt_lon": None,
            "retrieval_calls": [{"main": [[0, 0.99]]}],
        }
    )
    return {"inputs": {"sources": [{"source_id": "main", "database": str(db)}]}, "rows": rows}


def test_scoring_and_suggestions(tmp_path):
    db = tmp_path / "db.h5"
    x0, y0 = _db(db)
    report = _report(db, x0, y0)
    recs = R.score_calls(report, {"main": R.load_footprints(db)})
    assert len(recs) == 120 and sum(r["first"] for r in recs) == 60
    firsts = [r for r in recs if r["first"]]
    assert sum(not r["correct"] for r in firsts) == 15  # rows 0, 4, 8, ...
    assert all(r["any_correct_topk"] for r in recs)
    res = R.analyse(recs, precision=1.0, keep=0.95, min_pass=30, n_boot=200)
    assert res["top1_correct_rate"] == pytest.approx(105 / 120)
    # wrong top-1 scores are <= 0.80, correct >= 0.85: the cut lands right above the
    # highest wrong score
    max_wrong = max(r["score"] for r in recs if not r["correct"])
    t_ro = res["suggested"]["retrieval_only_min_score"]["t"]
    assert max_wrong < t_ro <= round(max_wrong + 0.01, 2) + 1e-9
    assert res["suggested"]["rescan_min_score"]["t"] >= 0.85
    cur = res["current"]["retrieval_only_min_score"]
    assert cur["precision"] == 1.0 and cur["precision_ci95"] is not None


def test_main_writes_json(tmp_path, capsys):
    db = tmp_path / "db.h5"
    x0, y0 = _db(db)
    rep = tmp_path / "r.json"
    rep.write_text(json.dumps(_report(db, x0, y0)), encoding="utf-8")
    out = tmp_path / "t.json"
    assert R.main([str(rep), "--out", str(out), "--bootstrap", "50"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["calls"] == 120 and data["curve_all_calls"][0]["t"] == 0.3
    empty = tmp_path / "e.json"
    empty.write_text(json.dumps({"inputs": {"sources": []}, "rows": []}), encoding="utf-8")
    assert R.main([str(empty)]) == 1
