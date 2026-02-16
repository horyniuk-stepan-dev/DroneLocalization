import numpy as np
import json
from src.geometry.transformations import GeometryTransforms
from src.geometry.coordinates import CoordinateConverter
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class GPSCalibration:
    """GPS calibration and coordinate transformation manager"""

    def __init__(self):
        self.affine_matrix = None
        self.is_calibrated = False
        logger.info("GPSCalibration initialized")

    def calibrate(self, points_2d: list, points_gps: list) -> dict:
        """Calculate affine binding matrix"""
        logger.info(f"Starting GPS calibration with {len(points_2d)} points")

        if len(points_2d) < 3 or len(points_gps) < 3:
            logger.error(f"Insufficient points for calibration: {len(points_2d)} 2D, {len(points_gps)} GPS")
            raise ValueError("Потрібно мінімум 3 точки для афінної трансформації")

        pts_2d_np = np.array(points_2d, dtype=np.float32)
        pts_metric = []

        logger.debug("Converting GPS coordinates to metric projection...")
        for i, (lat, lon) in enumerate(points_gps):
            x, y = CoordinateConverter.gps_to_metric(lat, lon)
            pts_metric.append((x, y))
            logger.debug(f"Point {i}: ({lat:.6f}, {lon:.6f}) -> ({x:.2f}, {y:.2f})")

        pts_metric_np = np.array(pts_metric, dtype=np.float32)

        logger.debug("Estimating affine transformation...")
        M, inliers = GeometryTransforms.estimate_affine(pts_2d_np, pts_metric_np)

        if M is None:
            logger.error("Failed to compute affine transformation")
            raise ValueError("Не вдалося обчислити афінну трансформацію")

        self.affine_matrix = M
        self.is_calibrated = True

        logger.info("Affine matrix computed successfully")
        logger.debug(f"Affine matrix:\n{M}")

        # Calculate RMSE
        transformed_metric = GeometryTransforms.apply_affine(pts_2d_np, self.affine_matrix)
        errors = np.linalg.norm(pts_metric_np - transformed_metric, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        inliers_count = int(np.sum(inliers)) if inliers is not None else len(points_2d)

        logger.success(f"GPS calibration completed: RMSE = {rmse:.4f} meters, {inliers_count} inliers")

        return {
            "status": "success",
            "rmse_meters": rmse,
            "inliers_count": inliers_count
        }

    def transform_to_gps(self, x_2d: float, y_2d: float) -> tuple:
        """Transform 2D coordinate to real GPS coordinates"""
        if not self.is_calibrated or self.affine_matrix is None:
            logger.error("Attempted to transform coordinates without calibration")
            raise RuntimeError("GPS калібрування не виконано")

        logger.debug(f"Transforming 2D point ({x_2d:.2f}, {y_2d:.2f}) to GPS")

        point_np = np.array([[x_2d, y_2d]], dtype=np.float32)
        metric_pt = GeometryTransforms.apply_affine(point_np, self.affine_matrix)[0]

        lat, lon = CoordinateConverter.metric_to_gps(metric_pt[0], metric_pt[1])

        logger.debug(f"Transformed to GPS: ({lat:.6f}, {lon:.6f})")
        return lat, lon

    def save(self, path: str):
        """Save calibration matrix to file"""
        if not self.is_calibrated:
            logger.error("Attempted to save calibration without calibration data")
            raise RuntimeError("Немає даних для збереження")

        logger.info(f"Saving calibration to: {path}")

        data = {
            "affine_matrix": self.affine_matrix.tolist()
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.success(f"Calibration saved successfully to {path}")

    def load(self, path: str):
        """Load calibration matrix from file"""
        logger.info(f"Loading calibration from: {path}")

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            self.affine_matrix = np.array(data["affine_matrix"], dtype=np.float32)
            self.is_calibrated = True

            logger.success(f"Calibration loaded successfully from {path}")
            logger.debug(f"Loaded affine matrix:\n{self.affine_matrix}")

        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            raise
