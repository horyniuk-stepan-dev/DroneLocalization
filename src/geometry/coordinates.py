import math
from pyproj import CRS, Transformer
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class CoordinateConverter:
    """Конвертація між системами координат за допомогою динамічної проєкції UTM"""

    _transformer_to_metric = None
    _transformer_to_gps = None
    _initialized = False

    @classmethod
    def _initialize_utm(cls, lat: float, lon: float):
        zone_number = int((lon + 180) / 6) + 1
        utm_crs = CRS(proj='utm', zone=zone_number, ellps='WGS84')
        wgs84_crs = CRS("EPSG:4326")

        cls._transformer_to_metric = Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)
        cls._transformer_to_gps = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)
        cls._initialized = True
        logger.info(f"Initialized UTM projection for zone {zone_number} based on ({lat:.4f}, {lon:.4f})")

    @staticmethod
    def gps_to_metric(lat: float, lon: float) -> tuple:
        if not CoordinateConverter._initialized:
            CoordinateConverter._initialize_utm(lat, lon)
        x, y = CoordinateConverter._transformer_to_metric.transform(lon, lat)
        return x, y

    @staticmethod
    def metric_to_gps(x: float, y: float) -> tuple:
        if not CoordinateConverter._initialized:
            raise ValueError("CoordinateConverter is not initialized.")
        lon, lat = CoordinateConverter._transformer_to_gps.transform(x, y)
        return lat, lon

    @staticmethod
    def haversine_distance(coord1: tuple, coord2: tuple) -> float:
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))