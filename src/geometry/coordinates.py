from pyproj import Transformer
import math

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CoordinateConverter:
    """Conversion between coordinate systems and distance calculations"""

    @staticmethod
    def gps_to_metric(lat: float, lon: float) -> tuple:
        """Convert GPS (EPSG:4326) to Web Mercator metric projection (EPSG:3857)"""
        logger.debug(f"Converting GPS to metric: ({lat:.6f}, {lon:.6f})")

        # always_xy=True ensures coordinate order (longitude, latitude) or (x, y)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, y = transformer.transform(lon, lat)

        logger.debug(f"Metric coordinates: ({x:.2f}, {y:.2f})")
        return x, y

    @staticmethod
    def metric_to_gps(x: float, y: float) -> tuple:
        """Reverse conversion from Web Mercator metric to GPS"""
        logger.debug(f"Converting metric to GPS: ({x:.2f}, {y:.2f})")

        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)

        logger.debug(f"GPS coordinates: ({lat:.6f}, {lon:.6f})")
        return lat, lon

    @staticmethod
    def haversine_distance(coord1: tuple, coord2: tuple) -> float:
        """Calculate distance between two GPS coordinates (lat, lon) in meters"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2

        logger.debug(f"Calculating haversine distance between ({lat1:.6f}, {lon1:.6f}) and ({lat2:.6f}, {lon2:.6f})")

        R = 6371000  # Earth radius in meters

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2.0) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * \
            math.sin(delta_lambda / 2.0) ** 2

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c
        logger.debug(f"Haversine distance: {distance:.2f} meters")

        return distance
