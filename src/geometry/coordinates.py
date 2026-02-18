import math
from pyproj import Transformer
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CoordinateConverter:
    """Конвертація між системами координат за допомогою надійної проєкції Web Mercator"""

    _gps_to_metric_transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:3857", always_xy=True
    )
    _metric_to_gps_transformer = Transformer.from_crs(
        "EPSG:3857", "EPSG:4326", always_xy=True
    )

    @staticmethod
    def gps_to_metric(lat: float, lon: float) -> tuple:
        """Перевід GPS у метри (EPSG:3857)"""
        x, y = CoordinateConverter._gps_to_metric_transformer.transform(lon, lat)
        return x, y

    @staticmethod
    def metric_to_gps(x: float, y: float) -> tuple:
        """Зворотний перевід з метрів у GPS"""
        lon, lat = CoordinateConverter._metric_to_gps_transformer.transform(x, y)
        return lat, lon

    @staticmethod
    def haversine_distance(coord1: tuple, coord2: tuple) -> float:
        """Розрахунок фізичної відстані між двома GPS точками в метрах"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        R = 6371000  # Радіус Землі

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))