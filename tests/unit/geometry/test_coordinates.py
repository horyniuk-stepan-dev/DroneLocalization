import pytest

from src.geometry.coordinates import CoordinateConverter


def test_coordinate_converter_initialization():
    converter = CoordinateConverter(mode="UTM", reference_gps=(48.0, 35.0))
    assert converter.reference_gps == (48.0, 35.0)
    assert converter._mode == "UTM"
    assert converter.is_initialized


def test_wgs84_to_local_and_back():
    lat, lon = 48.01, 35.01
    converter = CoordinateConverter(mode="UTM", reference_gps=(48.0, 35.0))

    x, y = converter.gps_to_metric(lat, lon)
    # The result should be in meters, and > 0 since it's slightly north-east
    assert isinstance(x, float)
    assert isinstance(y, float)

    back_lat, back_lon = converter.metric_to_gps(x, y)
    assert pytest.approx(lat, abs=1e-6) == back_lat
    assert pytest.approx(lon, abs=1e-6) == back_lon


def test_distance_calculation():
    lat1, lon1 = 48.0, 35.0
    lat2, lon2 = 48.01, 35.0

    # Approx 1 degree of latitude is ~111km. So 0.01 degree is ~1.11km = 1110m
    dist = CoordinateConverter.haversine_distance((lat1, lon1), (lat2, lon2))
    assert 1100 < dist < 1120
