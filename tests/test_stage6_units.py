"""Етап 6 — одиниці (cos lat) та дрібне: mercator_scale_factor,
CoordinateConverter.ground_scale_factor, affine_fit_residual (якість фіту),
ground-scale у звіті валідатора, видалений мертвий ключ anchor_weight.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.graph import GraphOptimizationConfig
from scripts.validate_vs_telemetry import compute_report, mercator_y_to_lat
from src.geometry.coordinates import CoordinateConverter, mercator_scale_factor
from src.geometry.pose_graph_optimizer import affine_fit_residual


def _affine(tx=0.0, ty=0.0, s=0.5):
    return np.array([[s, 0.0, tx], [0.0, -s, ty]], dtype=np.float64)


# ── 6.1 cos(lat) ─────────────────────────────────────────────────────────────


def test_mercator_scale_factor():
    assert mercator_scale_factor(0.0) == pytest.approx(1.0)
    assert mercator_scale_factor(60.0) == pytest.approx(0.5, abs=1e-9)
    assert mercator_scale_factor(48.0) == pytest.approx(0.66913, abs=1e-4)


def test_ground_scale_factor_converter():
    assert CoordinateConverter("UTM", (48.0, 26.0)).ground_scale_factor() == 1.0
    wm = CoordinateConverter("WEB_MERCATOR")
    assert wm.ground_scale_factor(48.0) == pytest.approx(0.66913, abs=1e-4)
    assert wm.ground_scale_factor(None) == 1.0  # широта невідома → без корекції


def test_mercator_y_to_lat_roundtrip():
    R = 6378137.0
    for lat in [0.0, 30.0, 48.36, 60.0]:
        y = R * np.arcsinh(np.tan(np.radians(lat)))  # сферичний forward
        assert mercator_y_to_lat(y) == pytest.approx(lat, abs=1e-6)


def test_validator_reports_true_meters():
    gt = {s: _affine(tx=1000.0 + 50.0 * s, ty=2000.0) for s in range(5)}
    pred = {s: _affine(tx=1000.0 + 50.0 * s + 10.0, ty=2000.0) for s in range(5)}
    rep = compute_report(pred, gt, 100, 100, ground_scale=mercator_scale_factor(48.0))
    assert rep["overall"]["median"] == pytest.approx(10.0)  # Mercator-метри
    assert rep["overall_ground_m"]["median"] == pytest.approx(10.0 * 0.66913, abs=1e-3)
    # UTM (scale=1) → без окремого ground-блоку
    assert "overall_ground_m" not in compute_report(pred, gt, 100, 100, ground_scale=1.0)


# ── 6.2 якість афінного фіту ─────────────────────────────────────────────────


def test_affine_fit_residual_zero_for_affine():
    H = np.array([[0.9, -0.1, 30.0], [0.1, 0.9, -20.0], [0.0, 0.0, 1.0]])
    assert affine_fit_residual(H, 1280, 720) == pytest.approx(0.0, abs=1e-6)


def test_affine_fit_residual_positive_for_perspective():
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [4e-4, 2e-4, 1.0]])  # нахил
    r = affine_fit_residual(H, 1280, 720)
    assert r is not None and r > 1.0  # відчутний неафінний залишок (px)


# ── 6.3 мертвий ключ видалено ────────────────────────────────────────────────


def test_anchor_weight_removed():
    g = GraphOptimizationConfig()
    assert not hasattr(g, "anchor_weight")
    # м'які якорі несуть власні ваги
    assert hasattr(g, "anchor_base_w") and hasattr(g, "anchor_sigma_floor_m")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
