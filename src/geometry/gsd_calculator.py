"""Ground Sample Distance calculator.

GSD = (altitude * sensor_width) / (focal_length * image_width_px)
Yields meters per pixel — direct physical scale.

Usage:
    gsd = GSDCalculator(altitude_m=100, focal_mm=13.2,
                        sensor_w_mm=8.8, image_w_px=4000)
    meters_per_pixel = gsd.gsd_m_per_px  # e.g., 0.022 m/px
"""

from dataclasses import dataclass

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class GSDCalculator:
    altitude_m: float  # flight altitude [m] (reference)
    focal_length_mm: float  # focal length [mm]
    sensor_width_mm: float  # sensor width [mm]
    image_width_px: int  # image width [pixels]

    @property
    def gsd_m_per_px(self) -> float:
        """Meters per pixel for current flight parameters."""
        if self.focal_length_mm <= 0 or self.image_width_px <= 0:
            return 0.0
        gsd = (self.altitude_m * self.sensor_width_mm) / (
            self.focal_length_mm * self.image_width_px
        )
        return gsd

    @property
    def px_per_meter(self) -> float:
        """Pixels per meter — inverse GSD."""
        gsd = self.gsd_m_per_px
        return 1.0 / gsd if gsd > 1e-9 else 0.0

    def scale_from_altitude(self, actual_altitude_m: float) -> float:
        """Scale factor relative to reference altitude.

        If drone flies lower than reference -> scale > 1 (more pixels per meter).
        If drone flies higher than reference -> scale < 1 (fewer pixels per meter).
        """
        if actual_altitude_m <= 0:
            return 1.0
        return self.altitude_m / actual_altitude_m

    def log_summary(self) -> None:
        gsd = self.gsd_m_per_px
        logger.info(
            f"GSD Configuration: altitude={self.altitude_m}m, "
            f"focal={self.focal_length_mm}mm, "
            f"sensor={self.sensor_width_mm}mm, "
            f"img_w={self.image_width_px}px"
        )
        logger.info(f"Resulting GSD: {gsd * 100:.2f} cm/px ({self.px_per_meter:.1f} px/m)")
