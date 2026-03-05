import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.coordinates import CoordinateConverter

def test_conversions():
    lat, lon = 50.4501, 30.5234  # Kyiv
    
    # Test UTM (Default)
    CoordinateConverter.set_projection_mode('UTM')
    x_utm, y_utm = CoordinateConverter.gps_to_metric(lat, lon)
    lat_utm, lon_utm = CoordinateConverter.metric_to_gps(x_utm, y_utm)
    assert abs(lat - lat_utm) < 1e-7
    assert abs(lon - lon_utm) < 1e-7

    # Test Web Mercator
    CoordinateConverter.set_projection_mode('WEB_MERCATOR')
    x_wm, y_wm = CoordinateConverter.gps_to_metric(lat, lon)
    lat_wm, lon_wm = CoordinateConverter.metric_to_gps(x_wm, y_wm)
    assert abs(lat - lat_wm) < 1e-7
    assert abs(lon - lon_wm) < 1e-7
    
    # Verify they are different
    assert x_utm != x_wm
