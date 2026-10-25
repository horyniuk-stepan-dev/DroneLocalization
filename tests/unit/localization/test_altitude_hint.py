"""localization.altitude_hint: layer scales from telemetry or the last fix's scale."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.localization.altitude_hint import (
    DEFAULT_FOOTPRINT_FACTOR,
    AltitudeHint,
    footprint_factor_from_settings,
)
from src.localization.rotation_selector import RotationSelector
from src.localization.scale_manager import ScaleManager

# Relief project layers: 500 m, 1000 m (main) and 2000 m; ground pixel sizes in m/px.
LAYERS = {"500m": 0.33, "main": 0.72, "2000m": 1.51}
WIDTH = 1280


def _hint(**kwargs) -> AltitudeHint:
    return AltitudeHint(enabled=True, **kwargs)


def _config(enabled=True, max_layers=2):
    return {"localization": {"altitude_hint": {"enabled": enabled, "max_layers": max_layers}}}


def test_disabled_gives_nothing():
    hint = AltitudeHint.from_config(_config(enabled=False))
    assert hint.query_gsd(500.0, WIDTH, 0.3) == (None, "off")
    assert hint.layer_scales(0.3, LAYERS) == []
    assert not hint.learn(500.0, WIDTH, 0.3)


def test_telemetry_first_then_visual_then_none():
    hint = _hint()
    gsd, source = hint.query_gsd(700.0, WIDTH, 0.9)
    assert source == "telemetry"
    assert gsd == pytest.approx(700.0 * DEFAULT_FOOTPRINT_FACTOR / WIDTH)
    assert hint.query_gsd(None, WIDTH, 0.9) == (0.9, "visual")
    assert hint.query_gsd(float("nan"), WIDTH, 0.9) == (0.9, "visual")
    assert hint.query_gsd(0.0, WIDTH, None) == (None, "none")


def test_layer_scales_nearest_first_and_limited():
    hint = _hint(max_layers=2)
    # Between the 500 m and 1000 m layers: both are hinted, the 2000 m layer is not.
    scales = hint.layer_scales(0.46, LAYERS)
    assert [sid for sid, _ in scales] == ["500m", "main"]
    assert scales[0][1] == pytest.approx(0.46 / 0.33)
    assert scales[1][1] == pytest.approx(0.46 / 0.72)
    # Unknown GSD and ratios a frame cannot be normalized to get no hint.
    assert hint.layer_scales(0.46, {"a": None, "b": 0.0, "far": 0.05}) == []
    assert hint.layer_scales(None, LAYERS) == []


def test_from_config_reads_settings_and_camera():
    settings = SimpleNamespace(focal_length_mm=20.0, sensor_width_mm=10.0)
    hint = AltitudeHint.from_config(_config(max_layers=3), settings)
    assert hint.enabled and hint.max_layers == 3
    assert hint.footprint_factor == pytest.approx(0.5)
    assert footprint_factor_from_settings(None) == pytest.approx(DEFAULT_FOOTPRINT_FACTOR)
    bad = SimpleNamespace(focal_length_mm=0.0, sensor_width_mm=8.8)
    assert footprint_factor_from_settings(bad) == pytest.approx(DEFAULT_FOOTPRINT_FACTOR)


def test_learning_refines_footprint_and_rejects_outliers():
    hint = _hint(ema=0.5)
    true_factor = 0.8
    visual = 500.0 * true_factor / WIDTH
    assert hint.learn(500.0, WIDTH, visual)
    assert hint.footprint_factor == pytest.approx(
        0.5 * DEFAULT_FOOTPRINT_FACTOR + 0.5 * true_factor
    )
    before = hint.footprint_factor
    # Ten times the camera model (wrong map units or a bad fix) is ignored, both ways.
    assert not hint.learn(500.0, WIDTH, 500.0 * 10 * DEFAULT_FOOTPRINT_FACTOR / WIDTH)
    assert not hint.learn(500.0, WIDTH, 500.0 * 0.1 * DEFAULT_FOOTPRINT_FACTOR / WIDTH)
    assert hint.footprint_factor == before


def test_scale_manager_hints_without_prior():
    sm = ScaleManager({})
    sm.set_altitude_scales([1.4, 0.64])
    assert sm.altitude_primary == pytest.approx(1.4)
    candidates = sm.candidates()
    assert candidates[:2] == [pytest.approx(1.4), pytest.approx(0.64)]
    # The pyramid follows without near-duplicates of the hints (0.7 ~ 0.64, 1.4 = 1.4).
    assert candidates[2:] == [0.5, 1.0, 2.0]


def test_scale_manager_prior_stays_first():
    sm = ScaleManager({})
    sm._prior = 1.0
    sm.set_altitude_scales([1.05, 0.5])
    assert sm.altitude_primary is None
    assert sm.candidates() == [1.0, 0.5]  # 1.05 is the prior's view
    sm.set_altitude_scales(None)
    assert sm.candidates() == [1.0]


def test_scale_manager_without_hint_is_unchanged():
    sm = ScaleManager({})
    sm.set_altitude_scales([float("nan"), -1.0, None])
    assert sm.altitude_primary is None
    assert sm.candidates() == [0.5, 0.7, 1.0, 1.4, 2.0]
    sm.set_altitude_scales([1.4])
    sm.reset()
    assert sm.candidates() == [0.5, 0.7, 1.0, 1.4, 2.0]


def test_cascade_stage_one_uses_the_hinted_scale():
    sm = ScaleManager({})
    sm.set_altitude_scales([1.4])
    stages = RotationSelector._plan_stages([0, 90], sm.candidates(), sm, use_cascade=True)
    assert stages[0] == [(0, pytest.approx(1.4)), (90, pytest.approx(1.4))]
    plain = ScaleManager({})
    stages = RotationSelector._plan_stages([0, 90], plain.candidates(), plain, use_cascade=True)
    assert stages[0] == [(0, 1.0), (90, 1.0)]
