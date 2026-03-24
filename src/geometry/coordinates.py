import math
import threading

from pyproj import CRS, Transformer

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CoordinateConverter:
    """Детермінована конвертація координат (WebMercator або UTM)"""

    _lock = threading.Lock()
    _transformer_to_metric = None
    _transformer_to_gps = None
    _initialized = False
    _projection_mode = "WEB_MERCATOR"
    _reference_gps = None

    @classmethod
    def reset(cls):
        """Скидає проєкцію при зміні проєкту/відеобази."""
        with cls._lock:
            cls._initialized = False
            cls._reference_gps = None
            cls._transformer_to_metric = None
            cls._transformer_to_gps = None
            logger.info("CoordinateConverter reset")

    @classmethod
    def configure_projection(cls, mode: str, reference_gps: tuple = None):
        """
        Явне налаштування проєкції для проєкту.
        mode: 'WEB_MERCATOR' (EPSG:3857) або 'UTM'
        reference_gps: (lat, lon) обов'язковий тільки для UTM
        """
        with cls._lock:
            mode = mode.upper()
            if mode not in ["UTM", "WEB_MERCATOR"]:
                raise ValueError(f"Unsupported projection mode: {mode}")

            cls._projection_mode = mode
            if reference_gps:
                cls._reference_gps = (float(reference_gps[0]), float(reference_gps[1]))
            else:
                cls._reference_gps = None

            # WEB_MERCATOR не потребує reference point
            if mode == "WEB_MERCATOR":
                cls._initialize_projection(0, 0)
            elif cls._reference_gps:
                cls._initialize_projection(*cls._reference_gps)
            else:
                logger.warning(
                    "UTM configuration called without reference_gps. Initialization deferred."
                )
                cls._initialized = False

            logger.info(f"CoordinateConverter configured: {mode} (ref={cls._reference_gps})")

    @classmethod
    def export_projection_metadata(cls) -> dict:
        """Експорт поточних налаштувань для серіалізації в JSON/HDF5"""
        return {"mode": cls._projection_mode, "reference_gps": cls._reference_gps}

    @classmethod
    def load_projection_metadata(cls, meta: dict):
        """Відновлення проєкції з метаданих"""
        if not meta:
            logger.warning("No projection metadata found, falling back to WEB_MERCATOR")
            cls.configure_projection("WEB_MERCATOR")
            return

        mode = meta.get("mode", "WEB_MERCATOR")
        ref = meta.get("reference_gps")
        cls.configure_projection(mode, tuple(ref) if ref else None)

    @classmethod
    def _initialize_projection(cls, lat: float, lon: float):
        wgs84_crs = CRS("EPSG:4326")

        if cls._projection_mode == "UTM":
            # Якщо референс не заданий явно, ініціалізуємо UTM по першій точці
            if cls._reference_gps is None:
                cls._reference_gps = (lat, lon)
                logger.warning(f"Auto-initializing UTM reference from point: {cls._reference_gps}")

            ref_lat, ref_lon = cls._reference_gps
            zone_number = int((ref_lon + 180) / 6) + 1
            target_crs = CRS(proj="utm", zone=zone_number, ellps="WGS84")
            logger.info(
                f"Initialized UTM projection for zone {zone_number} based on ({ref_lat:.4f}, {ref_lon:.4f})"
            )
        else:
            target_crs = CRS("EPSG:3857")
            logger.info("Initialized WEB_MERCATOR projection (EPSG:3857)")

        cls._transformer_to_metric = Transformer.from_crs(wgs84_crs, target_crs, always_xy=True)
        cls._transformer_to_gps = Transformer.from_crs(target_crs, wgs84_crs, always_xy=True)
        cls._initialized = True

    @staticmethod
    def gps_to_metric(lat: float, lon: float) -> tuple:
        with CoordinateConverter._lock:
            if not CoordinateConverter._initialized:
                # Fallback для WebMercator, якщо не було конфігурації
                if CoordinateConverter._projection_mode == "WEB_MERCATOR":
                    CoordinateConverter._initialize_projection(lat, lon)
                else:
                    raise RuntimeError(
                        "CoordinateConverter (UTM) must be configured with reference_gps before use."
                    )

            x, y = CoordinateConverter._transformer_to_metric.transform(lon, lat)
            return x, y

    @staticmethod
    def metric_to_gps(x: float, y: float) -> tuple:
        with CoordinateConverter._lock:
            if not CoordinateConverter._initialized:
                if CoordinateConverter._projection_mode == "WEB_MERCATOR":
                    CoordinateConverter._initialize_projection(0, 0)
                else:
                    raise RuntimeError("CoordinateConverter is not initialized.")

            lon, lat = CoordinateConverter._transformer_to_gps.transform(x, y)
            return lat, lon

    @staticmethod
    def haversine_distance(coord1: tuple, coord2: tuple) -> float:
        """Розрахунок фізичної відстані між двома GPS точками в метрах"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        R = 6371000  # Радіус Землі

        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
