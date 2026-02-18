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
        self.calib_frame_id = None
        logger.info("GPSCalibration initialized")

    def calibrate(self, points_2d: list, points_gps: list, calib_frame_id: int = 0) -> dict:
        """Calculate affine binding matrix for specific calibration frame"""
        logger.info(f"Starting GPS calibration with {len(points_2d)} points, calib_frame_id={calib_frame_id}")

        if len(points_2d) < 3 or len(points_gps) < 3:
            raise ValueError("Потрібно мінімум 3 точки для афінної трансформації")

        pts_2d_np = np.array(points_2d, dtype=np.float32)
        pts_metric = []

        for i, (lat, lon) in enumerate(points_gps):
            x, y = CoordinateConverter.gps_to_metric(lat, lon)
            pts_metric.append((x, y))
            logger.debug(f"Point {i}: ({lat:.6f}, {lon:.6f}) -> ({x:.2f}, {y:.2f})")

        pts_metric_np = np.array(pts_metric, dtype=np.float32)
        M, inliers = GeometryTransforms.estimate_affine(pts_2d_np, pts_metric_np)

        if M is None:
            raise ValueError("Не вдалося обчислити афінну трансформацію")

        self.affine_matrix = M
        self.calib_frame_id = calib_frame_id
        self.is_calibrated = True

        transformed_metric = GeometryTransforms.apply_affine(pts_2d_np, self.affine_matrix)
        errors = np.linalg.norm(pts_metric_np - transformed_metric, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        inliers_count = int(np.sum(inliers)) if inliers is not None else len(points_2d)

        logger.success(
            f"GPS calibration completed: RMSE={rmse:.4f}m, "
            f"{inliers_count} inliers, calib_frame_id={calib_frame_id}"
        )
        return {"status": "success", "rmse_meters": rmse,
                "inliers_count": inliers_count, "calib_frame_id": calib_frame_id}

    def pixel_to_metric(self, x_2d: float, y_2d: float) -> tuple:
        """
        Transform pixel coordinate (in calib_frame space) → metric (EPSG:3857).
        Input MUST already be in calib_frame pixel space.
        Localizer handles H(best_match → calib_frame) before calling this.
        """
        if not self.is_calibrated or self.affine_matrix is None:
            raise RuntimeError("GPS калібрування не виконано")

        point_np = np.array([[x_2d, y_2d]], dtype=np.float32)
        metric_pt = GeometryTransforms.apply_affine(point_np, self.affine_matrix)[0]
        logger.debug(f"pixel ({x_2d:.2f}, {y_2d:.2f}) -> metric ({metric_pt[0]:.2f}, {metric_pt[1]:.2f})")
        return float(metric_pt[0]), float(metric_pt[1])

    def transform_to_gps(self, x_2d: float, y_2d: float) -> tuple:
        """
        Transform pixel coordinate (in calib_frame space) → GPS (lat, lon).
        ВИПРАВЛЕНО: метод відновлено для сумісності з main_window.py.
        Внутрішньо використовує pixel_to_metric + metric_to_gps.
        """
        if not self.is_calibrated or self.affine_matrix is None:
            raise RuntimeError("GPS калібрування не виконано")

        mx, my = self.pixel_to_metric(x_2d, y_2d)
        lat, lon = CoordinateConverter.metric_to_gps(mx, my)
        logger.debug(f"transform_to_gps: pixel ({x_2d:.2f}, {y_2d:.2f}) -> GPS ({lat:.6f}, {lon:.6f})")
        return lat, lon

    def save(self, path: str):
        """Save calibration to JSON file"""
        if not self.is_calibrated:
            raise RuntimeError("Немає даних для збереження")

        data = {
            "affine_matrix": self.affine_matrix.tolist(),
            "calib_frame_id": int(self.calib_frame_id) if self.calib_frame_id is not None else 0
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.success(f"Calibration saved: {path} (calib_frame_id={self.calib_frame_id})")

    def load(self, path: str):
        """Load calibration from JSON file"""
        logger.info(f"Loading calibration from: {path}")
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            self.affine_matrix = np.array(data["affine_matrix"], dtype=np.float32)

            if "calib_frame_id" in data:
                self.calib_frame_id = int(data["calib_frame_id"])
            else:
                self.calib_frame_id = 0
                logger.warning(
                    "Файл калібрування не містить calib_frame_id (старий формат). "
                    "Використовується frame_id=0. "
                    "Рекомендується перекалібрувати для точної локалізації."
                )

            self.is_calibrated = True
            logger.success(f"Calibration loaded: {path} (calib_frame_id={self.calib_frame_id})")

        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            raise