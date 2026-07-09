"""Юніт-тести чистих функцій GT-валідатора (scripts/validate_vs_telemetry.py).

Покривають метрики, інтерполяцію телеметрії та розрізи слотів — без h5py/pyproj,
тож бігають у пісочниці. Орієнтир: pred==gt → нульова похибка; відомий зсув →
median≈зсув; дуги/прямі/near-anchor коректно класифікуються.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_vs_telemetry import (  # noqa: E402
    affine_angle_deg,
    classify_slots,
    compute_report,
    frame_center,
    interp_telemetry,
    iso_scale,
    summarize,
)


def _affine(tx=0.0, ty=0.0, s=0.5, angle_deg=0.0, flip=True):
    a = np.radians(angle_deg)
    sign = -1.0 if flip else 1.0
    c, si = np.cos(a), np.sin(a)
    return np.array([[c * s, -si * sign * s, tx], [si * s, c * sign * s, ty]], dtype=np.float64)


def test_frame_center_and_scale():
    M = _affine(tx=100.0, ty=200.0, s=0.5, flip=True)
    c = frame_center(M, 10.0, 20.0)
    # center = R@[10,20] + t ; with flip the y-scale is negative
    assert c[0] == pytest.approx(100.0 + 0.5 * 10.0)
    assert c[1] == pytest.approx(200.0 - 0.5 * 20.0)
    assert iso_scale(M) == pytest.approx(0.5, rel=1e-9)


def test_affine_angle():
    assert affine_angle_deg(_affine(angle_deg=0.0)) == pytest.approx(0.0, abs=1e-9)
    assert affine_angle_deg(_affine(angle_deg=30.0)) == pytest.approx(30.0, abs=1e-6)


def test_summarize_empty_and_basic():
    e = summarize(np.array([]))
    assert e["n"] == 0 and e["median"] == 0.0
    d = summarize(np.array([1.0, 2.0, 3.0, 100.0]))
    assert d["n"] == 4 and d["median"] == pytest.approx(2.5) and d["max"] == 100.0


def test_interp_telemetry_in_and_out_of_range():
    t = np.array([0.0, 1.0, 2.0])
    x = np.array([0.0, 10.0, 20.0])
    y = np.array([0.0, -5.0, -10.0])
    out = interp_telemetry(t, x, y, np.array([0.5, 1.5, 3.0]))
    assert out[0] == pytest.approx([5.0, -2.5])
    assert out[1] == pytest.approx([15.0, -7.5])
    assert np.all(np.isnan(out[2]))  # поза діапазоном → NaN, без екстраполяції


def test_classify_slots_turns_and_anchors():
    slots = np.arange(10)
    # прямо 0..4, розворот 5..7, прямо 8..9
    headings = np.array([0, 0, 0, 0, 0, 20, 60, 90, 90, 90], dtype=float)
    m = classify_slots(slots, headings, anchor_slots={0, 9},
                       turn_rate_deg_per_slot=3.0, near_anchor_k=1)
    assert m["turn_arcs"].sum() >= 3           # дуга спіймана
    assert m["straights"][0] and m["straights"][2]
    assert m["near_anchor"][0] and m["near_anchor"][1] and m["near_anchor"][9]
    assert not m["near_anchor"][5]
    assert m["mid_leg"][5] and not m["mid_leg"][0]


def test_classify_no_headings_all_straight():
    m = classify_slots(np.arange(5), None, anchor_slots=set())
    assert m["straights"].all() and not m["turn_arcs"].any()


def test_compute_report_perfect_and_offset():
    # GT: 6 слотів на прямій (const heading) з рухомим центром
    gt = {s: _affine(tx=1000.0 + 50.0 * s, ty=2000.0, s=0.5) for s in range(6)}
    headings = {s: 0.0 for s in range(6)}

    perfect = compute_report(gt, gt, 100, 100, headings_deg=headings,
                             anchor_slots={0, 5}, near_anchor_k=1)
    assert perfect["overall"]["median"] == pytest.approx(0.0, abs=1e-9)
    assert perfect["overall"]["angle_deg_median"] == pytest.approx(0.0, abs=1e-9)

    # зсув усіх передбачень на 4 м по x
    pred = {s: _affine(tx=1000.0 + 50.0 * s + 4.0, ty=2000.0, s=0.5) for s in range(6)}
    rep = compute_report(pred, gt, 100, 100, headings_deg=headings,
                         anchor_slots={0, 5}, near_anchor_k=1)
    assert rep["overall"]["median"] == pytest.approx(4.0, abs=1e-9)
    assert rep["cuts"]["straights"]["n"] == 6
    assert rep["coverage"] == pytest.approx(1.0)


def test_compute_report_scale_and_angle_error():
    gt = {s: _affine(tx=0.0, ty=0.0, s=0.5, angle_deg=0.0) for s in range(4)}
    pred = {s: _affine(tx=0.0, ty=0.0, s=0.55, angle_deg=5.0) for s in range(4)}
    rep = compute_report(pred, gt, 100, 100)
    assert rep["overall"]["scale_rel_median"] == pytest.approx(0.1, abs=1e-6)  # 0.55/0.5-1
    assert rep["overall"]["angle_deg_median"] == pytest.approx(5.0, abs=1e-4)


def test_compute_report_no_overlap_raises():
    gt = {0: _affine()}
    pred = {99: _affine()}
    with pytest.raises(ValueError):
        compute_report(pred, gt, 100, 100)


def test_telemetry_cross_check():
    gt = {s: _affine(tx=100.0 * s, ty=0.0, s=0.5) for s in range(4)}
    pred = gt
    tel = {s: frame_center(gt[s], 50.0, 50.0) + np.array([2.0, 0.0]) for s in range(4)}
    rep = compute_report(pred, gt, 100, 100, telemetry_centers=tel)
    assert rep["telemetry_cross_check"]["median"] == pytest.approx(2.0, abs=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
