"""Запобіжники temporal-VO (Етап 8, сесія 2026-07-12) — vo_guards.

Сценарій із lasttest: аліасинг на повторюваній ріллі дає КОНСИСТЕНТНО хибні
temporal-ребра (усі брешуть однаково) → резидуали малі, а траєкторія на км
убік. Перевіряємо: (1) санітарний гейт одного ребра; (2) звірку проміжків
між якорями (ok / inconsistent / broken); (3) приглушення ребер проміжку;
(4) відбір кадрів на інтерполяційне перезаповнення.
"""

import numpy as np

from src.geometry.pose_graph.model_5dof import _state_to_affine
from src.geometry.pose_graph.vo_guards import (
    check_anchor_gaps,
    downweight_gap_edges,
    select_gap_fallback_frames,
    temporal_edge_sane,
)
from src.geometry.pose_graph_optimizer import PoseGraphOptimizer

CX, CY = 640.0, 360.0
W, H = 1280, 720


def _anchor_affine(cx_m, cy_m):
    return _state_to_affine(np.array([cx_m, cy_m, 0.0, 0.0, 0.0]), CX, CY, sign=1.0)


def _step(dx, dy=0.0):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)


def _grid_graph(poison=None, drop_edge=None, n=11, step=100.0):
    """Ланцюг 0..n-1 із кроком +step по X; якорі на 0 і n-1.

    poison: dict {from_id: (dx, dy)} — хибний вимір ребра (аліасинг);
    drop_edge: from_id ребра, якого немає (розрив ланцюга).
    """
    opt = PoseGraphOptimizer(W, H)
    for i in range(n):
        opt.add_node(i)
    for i in range(n - 1):
        if drop_edge is not None and i == drop_edge:
            continue
        dx, dy = (poison or {}).get(i, (step, 0.0))
        opt.add_edge(i, i + 1, _step(dx, dy), weight=1.0, edge_type="temporal")
    opt.fix_node(0, _anchor_affine(0.0, 0.0))
    opt.fix_node(n - 1, _anchor_affine((n - 1) * step, 0.0))
    return opt


# ── temporal_edge_sane ───────────────────────────────────────────────────────


def test_sane_edge_passes():
    ok, reason = temporal_edge_sane(_step(0.0, 300.0), 1, W, H)
    assert ok, reason


def test_wild_rotation_rejected():
    ang = np.radians(45.0)
    M = np.array(
        [[np.cos(ang), -np.sin(ang), 0.0], [np.sin(ang), np.cos(ang), 0.0]], dtype=np.float64
    )
    ok, reason = temporal_edge_sane(M, 1, W, H, max_rotation_deg=30.0)
    assert not ok and "поворот" in reason


def test_wild_scale_rejected():
    # масштаб 0.7 за межею 1/1.4 — реальний кейс: дегенеративна H від 4 інлаєрів
    M = np.array([[0.7, 0.0, 0.0], [0.0, 0.7, 100.0]], dtype=np.float64)
    ok, reason = temporal_edge_sane(M, 1, W, H, max_scale_ratio=1.4)
    assert not ok and "масштаб" in reason


def test_wild_shift_rejected_but_scales_with_gap():
    diag = float(np.hypot(W, H))
    M = _step(0.0, 1.3 * diag)
    ok, _ = temporal_edge_sane(M, 1, W, H, max_shift_frac=1.2)
    assert not ok
    ok, _ = temporal_edge_sane(M, 2, W, H, max_shift_frac=1.2)  # gap=2 → межа ×2
    assert ok


# ── check_anchor_gaps ────────────────────────────────────────────────────────


def test_consistent_gap_ok():
    opt = _grid_graph()
    rep = check_anchor_gaps(opt.edges, opt.anchor_states(), opt.sign, max_dev_m=50.0)
    assert rep[(0, 10)]["status"] == "ok"
    assert rep[(0, 10)]["dev_m"] < 1e-6


def test_aliased_gap_inconsistent():
    # аліасинг: ребра 3..7 однаково брешуть (зсув назад) — як кадри 1–21 lasttest
    poison = {i: (-150.0, 80.0) for i in range(3, 8)}
    opt = _grid_graph(poison=poison)
    rep = check_anchor_gaps(opt.edges, opt.anchor_states(), opt.sign, max_dev_m=150.0)
    assert rep[(0, 10)]["status"] == "inconsistent"
    assert rep[(0, 10)]["dev_m"] > 1000.0


def test_broken_gap_detected():
    opt = _grid_graph(drop_edge=5)
    rep = check_anchor_gaps(opt.edges, opt.anchor_states(), opt.sign, max_dev_m=50.0)
    assert rep[(0, 10)]["status"] == "broken"


def test_downweight_only_gap_edges():
    opt = _grid_graph()
    n = downweight_gap_edges(opt.edges, [(0, 10)], factor=0.05)
    assert n == 10
    assert all(abs(e.weight - 0.05) < 1e-12 for e in opt.edges)
    # порожній список проміжків → no-op
    assert downweight_gap_edges(opt.edges, [], factor=0.05) == 0


# ── select_gap_fallback_frames ───────────────────────────────────────────────


def test_fallback_selects_only_deviant_frames():
    opt = _grid_graph()
    anchor_states = opt.anchor_states()
    # кадри 1,2,8,9 на лінії; 4..6 віднесло на 900 м (бульбашка 297–305 lasttest)
    centers = {f: (f * 100.0, 0.0) for f in range(1, 10)}
    for f in (4, 5, 6):
        centers[f] = (f * 100.0 - 600.0, 700.0)
    sel = select_gap_fallback_frames(centers, anchor_states, [(0, 10)], max_dev_m=150.0)
    assert sel == {4, 5, 6}


def test_fallback_ignores_missing_and_small_dev():
    opt = _grid_graph()
    anchor_states = opt.anchor_states()
    centers = {f: (f * 100.0 + 30.0, -40.0) for f in range(1, 10)}  # відхилення 50 м
    del centers[5]  # кадру без результату немає в кандидатах
    sel = select_gap_fallback_frames(centers, anchor_states, [(0, 10)], max_dev_m=150.0)
    assert sel == set()
