import math
from typing import Any

from pyproj import CRS, Transformer

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CoordinateConverter:
    """Детермінована конвертація координат (WebMercator або UTM) на основі екземпляра."""

    def __init__(
        self, mode: str = "WEB_MERCATOR", reference_gps: tuple[float, float] | None = None
    ):
        self._mode = mode.upper()
        self._reference_gps = reference_gps
        self._transformer_to_metric: Transformer | None = None
        self._transformer_to_gps: Transformer | None = None
        self._initialized = False

        if self._mode == "WEB_MERCATOR":
            self._initialize_projection(0.0, 0.0)
        elif self._reference_gps:
            self._initialize_projection(*self._reference_gps)

    @property
    def is_initialized(self) -> bool:
        """Повертає True, якщо проєкція успішно ініціалізована."""
        return self._initialized

    @property
    def reference_gps(self) -> tuple[float, float] | None:
        """Повертає опорні GPS-координати, використані для UTM проєкції."""
        return self._reference_gps

    def _initialize_projection(self, lat: float, lon: float) -> None:
        wgs84_crs = CRS("EPSG:4326")

        if self._mode == "UTM":
            if self._reference_gps is None:
                self._reference_gps = (lat, lon)
                logger.warning(f"Auto-initializing UTM reference from point: {self._reference_gps}")

            ref_lat, ref_lon = self._reference_gps
            zone_number = int((ref_lon + 180) / 6) + 1
            target_crs = CRS(proj="utm", zone=zone_number, ellps="WGS84")
            logger.info(
                f"Initialized UTM projection for zone {zone_number} based on ({ref_lat:.4f}, {ref_lon:.4f})"
            )
        else:
            target_crs = CRS("EPSG:3857")
            logger.info("Initialized WEB_MERCATOR projection (EPSG:3857)")

        self._transformer_to_metric = Transformer.from_crs(wgs84_crs, target_crs, always_xy=True)
        self._transformer_to_gps = Transformer.from_crs(target_crs, wgs84_crs, always_xy=True)
        self._initialized = True

    def gps_to_metric(self, lat: float, lon: float) -> tuple[float, float]:
        if not self._initialized:
            if self._mode == "WEB_MERCATOR":
                self._initialize_projection(lat, lon)
            else:
                raise RuntimeError(
                    f"CoordinateConverter (UTM) must be initialized with reference_gps "
                    f"before converting ({lat}, {lon}). "
                    f"Call __init__ with reference_gps parameter first."
                )

        if self._transformer_to_metric is None:
            raise RuntimeError(
                f"GPS-to-metric transformer not initialized (mode={self._mode}). "
                f"Cannot convert ({lat}, {lon}). This is a bug — _initialize_projection should have been called."
            )

        x, y = self._transformer_to_metric.transform(lon, lat)
        return float(x), float(y)

    def metric_to_gps(self, x: float, y: float) -> tuple[float, float]:
        if not self._initialized:
            if self._mode == "WEB_MERCATOR":
                self._initialize_projection(0.0, 0.0)
            else:
                raise RuntimeError("CoordinateConverter is not initialized.")

        if self._transformer_to_gps is None:
            raise RuntimeError(
                f"Metric-to-GPS transformer not initialized (mode={self._mode}). "
                f"Cannot convert ({x}, {y}). This is a bug — _initialize_projection should have been called."
            )

        lon, lat = self._transformer_to_gps.transform(x, y)
        return float(lat), float(lon)

    def export_metadata(self) -> dict[str, Any]:
        """Експорт налаштувань для серіалізації."""
        return {"mode": self._mode, "reference_gps": self._reference_gps}

    @classmethod
    def from_metadata(cls, meta: dict[str, Any]) -> "CoordinateConverter":
        """Створення конвертера з метаданих."""
        if not meta:
            return cls("WEB_MERCATOR")
        mode = meta.get("mode", "WEB_MERCATOR")
        ref = meta.get("reference_gps")
        return cls(mode, tuple(ref) if ref else None)

    @staticmethod
    def haversine_distance(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
        """Розрахунок фізичної відстані між двома GPS точками в метрах."""
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


# Глобальний екземпляр для зворотної сумісності (тимчасово)
DEFAULT_CONVERTER = CoordinateConverter("WEB_MERCATOR")
