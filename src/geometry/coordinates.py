import math
from typing import Any

from pyproj import CRS, Transformer

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def mercator_scale_factor(lat: float) -> float:
    """Multiplier for Web-Mercator-meters → real ground meters: cos(lat).

    WebMercator (EPSG:3857) stretches distances by 1/cos(lat) (~1.5x at 48°), so
    metric figures in reports/GSD on real data are systematically overestimated.
    Multiplying Mercator distance by cos(lat) yields real meters. For UTM, correction = 1.0.
    On synthetic data (all in Mercator, relative comparison) does not affect conclusions.
    """
    return math.cos(math.radians(lat))


class CoordinateConverter:
    """Deterministic coordinate conversion (WebMercator or UTM) based on instance configuration."""

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
        """Returns True if the projection is successfully initialized."""
        return self._initialized

    @property
    def reference_gps(self) -> tuple[float, float] | None:
        """Returns reference GPS coordinates used for UTM projection."""
        return self._reference_gps

    @property
    def mode(self) -> str:
        """Projection mode: 'UTM' or 'WEB_MERCATOR' (public access instead of _mode)."""
        return self._mode

    def ground_scale_factor(self, lat: float | None = None) -> float:
        """Multiplier for 'projection meters -> real ground meters'.

        UTM -> 1.0 (already real meters). WEB_MERCATOR -> cos(lat): lat is taken from
        argument or reference_gps. If latitude is unknown -> 1.0 (no correction).
        """
        if self._mode != "WEB_MERCATOR":
            return 1.0
        if lat is None and self._reference_gps is not None:
            lat = self._reference_gps[0]
        if lat is None:
            return 1.0
        return mercator_scale_factor(lat)

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
        """Export settings for serialization."""
        return {"mode": self._mode, "reference_gps": self._reference_gps}

    @classmethod
    def from_metadata(cls, meta: dict[str, Any]) -> "CoordinateConverter":
        """Create a converter from metadata."""
        if not meta:
            return cls("WEB_MERCATOR")
        mode = meta.get("mode", "WEB_MERCATOR")
        ref = meta.get("reference_gps")
        return cls(mode, tuple(ref) if ref else None)

    @staticmethod
    def haversine_distance(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
        """Calculate physical distance between two GPS points in meters."""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        R = 6371000  # Earth radius in meters

        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Global instance for backward compatibility (temporary)
DEFAULT_CONVERTER = CoordinateConverter("WEB_MERCATOR")
