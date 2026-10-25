"""Live MAVLink flight data (src/flight_data.MavlinkFlightData)."""

from __future__ import annotations

import socket
import time

import pytest

from src.flight_data import FlightPrior, MavlinkFlightData, build_flight_prior


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _feed(src: MavlinkFlightData, t: float, yaw_rad: float, lat: float, agl: float) -> None:
    src.ingest("ATTITUDE", {"yaw": yaw_rad, "pitch": -0.1, "roll": 0.05}, t)
    src.ingest(
        "GLOBAL_POSITION_INT",
        {"lat": int(lat * 1e7), "lon": int(26.065 * 1e7), "alt": 612345, "hdg": 65535},
        t,
    )
    src.ingest("TERRAIN_REPORT", {"current_height": agl}, t)


def test_latest_state_and_interpolation():
    clock = FakeClock(10.0)
    src = MavlinkFlightData(clock=clock, autostart=False)
    assert src.sample_at(0.0) is None  # nothing received yet
    _feed(src, 10.0, yaw_rad=0.0, lat=48.0, agl=500.0)
    _feed(src, 10.2, yaw_rad=0.2, lat=48.001, agl=502.0)
    clock.t = 10.2
    s = src.sample_at(123.0)  # the argument's origin is ignored for live data
    assert s.time_s == 123.0
    assert s.heading_deg == pytest.approx(11.459, abs=1e-3)
    assert s.lat == pytest.approx(48.001)
    assert s.lon == pytest.approx(26.065)
    assert s.alt_msl_m == pytest.approx(612.345)
    assert s.alt_agl_m == pytest.approx(502.0)
    assert s.pitch_deg == pytest.approx(-5.7296, abs=1e-3)
    assert s.roll_deg == pytest.approx(2.8648, abs=1e-3)

    src.time_offset_s = -0.1  # video latency: state 100 ms ago, halfway between
    s = src.sample_at(0.0)
    assert s.lat == pytest.approx(48.0005)
    assert s.alt_agl_m == pytest.approx(501.0)


def test_heading_wraps_through_north():
    clock = FakeClock(1.0)
    src = MavlinkFlightData(clock=clock, autostart=False, time_offset_s=-0.5)
    src.ingest("ATTITUDE", {"yaw": -0.1, "pitch": 0.0, "roll": 0.0}, 0.0)  # 354.27 deg
    src.ingest("ATTITUDE", {"yaw": 0.1, "pitch": 0.0, "roll": 0.0}, 1.0)  # 5.73 deg
    assert src.sample_at(0.0).heading_deg == pytest.approx(0.0, abs=1e-6)


def test_stale_data_is_not_used():
    clock = FakeClock(5.0)
    src = MavlinkFlightData(clock=clock, autostart=False, max_gap_s=1.0)
    _feed(src, 5.0, yaw_rad=1.0, lat=48.0, agl=500.0)
    clock.t = 7.0
    assert src.sample_at(0.0) is None


def test_gps_heading_only_without_attitude():
    src = MavlinkFlightData(clock=FakeClock(0.0), autostart=False)
    src.ingest("GLOBAL_POSITION_INT", {"lat": 0, "lon": 0, "alt": 0, "hdg": 9000}, 0.0)
    assert src.sample_at(0.0).heading_deg == pytest.approx(90.0)
    src.ingest("ATTITUDE", {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}, 0.0)
    src.ingest("GLOBAL_POSITION_INT", {"lat": 0, "lon": 0, "alt": 0, "hdg": 9000}, 0.0)
    assert src.sample_at(0.0).heading_deg == pytest.approx(0.0)  # ATTITUDE wins


def test_downward_rangefinder_is_agl_other_orientations_ignored():
    src = MavlinkFlightData(clock=FakeClock(0.0), autostart=False)
    src.ingest("DISTANCE_SENSOR", {"orientation": 0, "current_distance": 1234}, 0.0)
    assert src.sample_at(0.0) is None
    src.ingest("DISTANCE_SENSOR", {"orientation": 25, "current_distance": 1234}, 0.0)
    assert src.sample_at(0.0).alt_agl_m == pytest.approx(12.34)


def test_build_flight_prior_mavlink_does_not_open_the_port():
    prior = build_flight_prior(
        {
            "flight_data": {
                "source": "mavlink",
                "mavlink_url": "udpin:127.0.0.1:1",
                "time_offset_s": -0.2,
            }
        }
    )
    assert isinstance(prior, FlightPrior)
    assert isinstance(prior.source, MavlinkFlightData)
    assert prior.source.time_offset_s == pytest.approx(-0.2)
    assert prior.source._conn is None
    prior.close()


def _free_udp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_udp_roundtrip_and_port_handover():
    mavlink = pytest.importorskip("pymavlink.dialects.v20.common")
    port = _free_udp_port()
    url = f"udpin:127.0.0.1:{port}"
    first = MavlinkFlightData(url=url)
    first.start()
    second = MavlinkFlightData(url=url)
    second.start()  # takes the port over from the first instance
    assert first._conn is None
    try:
        enc = mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
        frames = [
            enc.attitude_encode(1000, 0.0, 0.0, 1.5707963, 0.0, 0.0, 0.0).pack(enc),
            enc.global_position_int_encode(
                1000, 483650000, 260650000, 612345, 500000, 0, 1500, 0, 9000
            ).pack(enc),
            enc.terrain_report_encode(483650000, 260650000, 0, 238.5, 501.25, 0, 1).pack(enc),
        ]
        out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sample = None
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            for frame in frames:
                out.sendto(frame, ("127.0.0.1", port))
            time.sleep(0.05)
            sample = second.sample_at(0.0)
            if sample is not None and sample.alt_agl_m is not None and sample.lat is not None:
                break
        out.close()
        assert sample is not None
        assert sample.heading_deg == pytest.approx(90.0, abs=1e-3)
        assert sample.lat == pytest.approx(48.365)
        assert sample.alt_agl_m == pytest.approx(501.25)
    finally:
        second.close()
        first.close()
    assert second._conn is None
