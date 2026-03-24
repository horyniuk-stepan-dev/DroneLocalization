import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sys
from pathlib import Path

from src.geometry.coordinates import CoordinateConverter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_conversions():
    # Reset initialization to clean state
    CoordinateConverter.set_projection_mode("UTM")

    lat, lon = 50.4501, 30.5234  # Kyiv

    # Test automatic UTM projection
    x_utm, y_utm = CoordinateConverter.gps_to_metric(lat, lon)

    # Now it should be initialized
    assert CoordinateConverter._initialized is True

    lat_utm, lon_utm = CoordinateConverter.metric_to_gps(x_utm, y_utm)
    assert abs(lat - lat_utm) < 1e-7
    assert abs(lon - lon_utm) < 1e-7

    # Test Web Mercator
    CoordinateConverter.set_projection_mode("WEB_MERCATOR")
    x_wm, y_wm = CoordinateConverter.gps_to_metric(lat, lon)

    assert CoordinateConverter._initialized is True

    lat_wm, lon_wm = CoordinateConverter.metric_to_gps(x_wm, y_wm)
    assert abs(lat - lat_wm) < 1e-7
    assert abs(lon - lon_wm) < 1e-7

    # Verify they are different
    assert x_utm != x_wm


def test_haversine_distance():
    # Kyiv
    kyiv_lat, kyiv_lon = 50.4501, 30.5234
    # Lviv
    lviv_lat, lviv_lon = 49.8397, 24.0297

    # Distance is approx 468 km
    distance = CoordinateConverter.haversine_distance((kyiv_lat, kyiv_lon), (lviv_lat, lviv_lon))
    assert 460000 < distance < 480000


def test_uninitialized_error():
    # Reset initialization strictly for testing error
    CoordinateConverter._initialized = False
    with pytest.raises(ValueError, match="CoordinateConverter is not initialized."):
        CoordinateConverter.metric_to_gps(100.0, 100.0)
