"""Flight-data altitude for localization.altitude_hint: above ground, else above take-off."""

from __future__ import annotations

import pytest

from config.app import AppConfig
from src.flight_data import (
    CsvFlightData,
    FlightPrior,
    MavlinkFlightData,
    build_flight_prior,
)


def _csv(tmp_path, text, name="t.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _config(**flight_data):
    cfg = AppConfig().model_dump()
    cfg["flight_data"].update(flight_data)
    return cfg


def test_altitude_hint_is_off_by_default_and_altitude_is_used_when_on():
    cfg = AppConfig().model_dump()
    assert cfg["localization"]["altitude_hint"]["enabled"] is False
    assert cfg["flight_data"]["use_altitude"] is True


def test_csv_prefers_height_above_ground(tmp_path):
    p = _csv(tmp_path, "time_s,alt_agl_m,alt_rel_m\n0,480,500\n1,520,540\n2,,700\n")
    prior = build_flight_prior(_config(source="csv", csv_path=str(p)))
    assert prior.altitude_m(0.5) == pytest.approx(500.0)  # AGL, interpolated
    assert CsvFlightData(p).sample_at(0.5).alt_rel_m == pytest.approx(520.0)


def test_csv_relative_height_without_ground_height(tmp_path):
    p = _csv(tmp_path, "time_s,alt_rel_m\n0,1000\n1,1000\n")
    prior = build_flight_prior(_config(source="csv", csv_path=str(p)))
    assert prior.altitude_m(0.5) == pytest.approx(1000.0)


def test_flightsim_height_above_base_plane(tmp_path):
    p = _csv(tmp_path, "frame_index,timestamp,alt_z,yaw_rad\n0,0.0,1500,0\n1,1.0,1500,0\n")
    prior = FlightPrior(CsvFlightData(p, preset="flightsim"), _config())
    assert prior.altitude_m(0.5) == pytest.approx(1500.0)


def test_no_altitude_when_disabled_missing_or_invalid(tmp_path):
    p = _csv(tmp_path, "time_s,heading_deg,alt_agl_m\n0,90,0\n1,90,-5\n")
    prior = build_flight_prior(_config(source="csv", csv_path=str(p)))
    assert prior.altitude_m(0.5) is None  # zero / negative height is no height
    assert prior.yaw_hint_deg(0.5) is not None
    q = _csv(tmp_path, "time_s,heading_deg,alt_agl_m\n0,90,500\n1,90,500\n", "q.csv")
    off = build_flight_prior(_config(source="csv", csv_path=str(q), use_altitude=False))
    assert off.altitude_m(0.5) is None
    assert build_flight_prior(_config(source="csv", csv_path=str(q))).altitude_m(9.0) is None


class _Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_mavlink_relative_altitude_until_ground_height_arrives():
    clock = _Clock(10.0)
    src = MavlinkFlightData(clock=clock, autostart=False)
    position = {"lat": 480000000, "lon": 260650000, "alt": 1612345, "relative_alt": 1000500}
    src.ingest("GLOBAL_POSITION_INT", position, 10.0)
    prior = FlightPrior(src, _config())
    assert src.sample_at(0.0).alt_rel_m == pytest.approx(1000.5)
    assert prior.altitude_m(0.0) == pytest.approx(1000.5)
    src.ingest("TERRAIN_REPORT", {"current_height": 980.0}, 10.0)
    assert prior.altitude_m(0.0) == pytest.approx(980.0)
