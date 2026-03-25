import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.geometry.coordinates import CoordinateConverter


def test_conversions():
    lat, lon = 50.4501, 30.5234  # Kyiv

    # UTM конвертація (екземпляр-based API)
    converter_utm = CoordinateConverter("UTM", reference_gps=(lat, lon))
    x_utm, y_utm = converter_utm.gps_to_metric(lat, lon)

    lat_utm, lon_utm = converter_utm.metric_to_gps(x_utm, y_utm)
    assert abs(lat - lat_utm) < 1e-7
    assert abs(lon - lon_utm) < 1e-7

    # Web Mercator конвертація
    converter_wm = CoordinateConverter("WEB_MERCATOR")
    x_wm, y_wm = converter_wm.gps_to_metric(lat, lon)

    lat_wm, lon_wm = converter_wm.metric_to_gps(x_wm, y_wm)
    assert abs(lat - lat_wm) < 1e-7
    assert abs(lon - lon_wm) < 1e-7

    # Координати повинні відрізнятися між проєкціями
    assert x_utm != x_wm


def test_haversine_distance():
    # Kyiv
    kyiv_lat, kyiv_lon = 50.4501, 30.5234
    # Lviv
    lviv_lat, lviv_lon = 49.8397, 24.0297

    # Відстань ~468 км
    distance = CoordinateConverter.haversine_distance((kyiv_lat, kyiv_lon), (lviv_lat, lviv_lon))
    assert 460000 < distance < 480000


def test_uninitialized_error():
    # UTM без reference_gps повинен кидати RuntimeError при metric_to_gps
    converter = CoordinateConverter("UTM")
    with pytest.raises(RuntimeError, match="CoordinateConverter"):
        converter.metric_to_gps(100.0, 100.0)
