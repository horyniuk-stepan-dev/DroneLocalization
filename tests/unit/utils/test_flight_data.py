"""Optional telemetry (src/flight_data): priors only, vision-only by default."""

from __future__ import annotations

import math

import pytest

from config.app import AppConfig
from src.flight_data import (
    CsvFlightData,
    FlightPrior,
    NullFlightData,
    build_flight_prior,
    heading_to_yaw_hint,
)


def test_default_is_vision_only():
    assert build_flight_prior(AppConfig().model_dump()) is None
    assert NullFlightData().sample_at(1.0) is None


@pytest.mark.parametrize(
    "heading,ref,offset,expected",
    [(0, 0, 0, 0), (90, 0, 0, 270), (270, 0, 0, 90), (100, 90, 0, 350), (45, 0, 10, 305)],
)
def test_heading_to_yaw_hint(heading, ref, offset, expected):
    assert heading_to_yaw_hint(heading, ref, offset) == pytest.approx(expected)


def test_hint_rotates_like_the_continuous_mode():
    """Drone flying east over north-up layers: the frame's top points east, so it
    must turn 90° clockwise (cv2 angle -90 = 270) to become north-up."""
    import cv2
    import numpy as np

    img = np.zeros((101, 101), np.uint8)
    img[0:10, 45:56] = 255  # marker at the TOP of the frame (= east for this drone)
    m = cv2.getRotationMatrix2D((50, 50), heading_to_yaw_hint(90.0), 1.0)
    out = cv2.warpAffine(img, m, (101, 101))
    ys, xs = np.nonzero(out > 128)
    assert xs.mean() > 85  # east ends up on the right of a north-up image


def _csv(tmp_path, text):
    p = tmp_path / "t.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_csv_interpolates_heading_across_north(tmp_path):
    p = _csv(tmp_path, "time_s,heading_deg,alt_agl_m\n0,350,100\n1,10,120\n")
    s = CsvFlightData(p).sample_at(0.5)
    assert s.heading_deg == pytest.approx(0.0, abs=1e-9) or s.heading_deg == pytest.approx(360.0)
    assert s.alt_agl_m == pytest.approx(110)


def test_csv_stale_gap_returns_none(tmp_path):
    p = _csv(tmp_path, "time_s,heading_deg\n0,10\n5,20\n")
    assert CsvFlightData(p, max_gap_s=1.0).sample_at(2.5) is None
    assert CsvFlightData(p, max_gap_s=1.0).sample_at(4.5) is not None


def test_flightsim_preset_converts_ccw_yaw_to_compass(tmp_path):
    # FlightSimulator: yaw_rad counter-clockwise from north (east = -pi/2)
    p = _csv(tmp_path, f"frame_index,timestamp,alt_z,yaw_rad\n0,0.0,1000,{-math.pi / 2}\n")
    s = CsvFlightData(p, preset="flightsim").sample_at(0.0)
    assert s.heading_deg == pytest.approx(90.0)
    assert s.alt_msl_m == pytest.approx(1000.0)


def test_prior_from_config(tmp_path):
    p = _csv(tmp_path, "time_s,heading_deg\n" + "".join(f"{t},90\n" for t in range(11)))
    cfg = AppConfig().model_dump()
    cfg["flight_data"].update(source="csv", csv_path=str(p), camera_yaw_offset_deg=5.0)
    prior = build_flight_prior(cfg)
    assert isinstance(prior, FlightPrior)
    assert prior.yaw_hint_deg(3.0) == pytest.approx(265.0)
    cfg["flight_data"]["use_heading"] = False
    assert build_flight_prior(cfg).yaw_hint_deg(3.0) is None


def test_fit_yaw_offset_recovers_mount_offset(tmp_path):
    from scripts.fit_yaw_offset import fit

    rows = "time_s,heading_deg\n" + "".join(f"{t},{(t * 7) % 360}\n" for t in range(40))
    tel = CsvFlightData(_csv(tmp_path, rows))
    measured = [(float(t), heading_to_yaw_hint((t * 7) % 360, 0.0, 12.0)) for t in range(40)]
    res = fit(measured, tel)
    assert ((res["as_logged"]["offset_deg"] - 12.0 + 180) % 360) - 180 == pytest.approx(0, abs=1e-6)
    assert res["as_logged"]["spread_deg"] < 1e-3 < res["mirrored"]["spread_deg"]
