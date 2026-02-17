import numpy as np
import cv2
from src.geometry.transformations import GeometryTransforms
from src.localization.matcher import FastRetrieval, FeatureMatcher
from src.tracking.kalman_filter import TrajectoryFilter
from src.tracking.outlier_detector import OutlierDetector
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Localizer:
    """Main drone localization pipeline in real-time"""

    def __init__(self, database, feature_extractor, matcher, calibration, config=None):
        self.database = database
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.calibration = calibration
        self.config = config or {}

        self.min_matches = self.config.get('localization', {}).get('min_matches', 15)
        self.ransac_thresh = self.config.get('localization', {}).get('ransac_threshold', 3.0)

        logger.info("Initializing Localizer...")
        logger.info(f"Config: min_matches={self.min_matches}, ransac_threshold={self.ransac_thresh}")

        # Initialize fast retrieval system
        global_descs = self.database.global_descriptors
        self.retrieval = FastRetrieval(global_descs)
        logger.info(f"FastRetrieval initialized with {len(global_descs)} reference frames")

        # Initialize tracking and filtering modules
        self.kalman_filter = TrajectoryFilter(
            process_noise=self.config.get('tracking', {}).get('kalman_process_noise', 0.1),
            measurement_noise=self.config.get('tracking', {}).get('kalman_measurement_noise', 10.0)
        )
        logger.info("Kalman filter initialized")

        self.outlier_detector = OutlierDetector(
            threshold_std=self.config.get('tracking', {}).get('outlier_threshold_std', 3.0)
        )
        logger.info("Outlier detector initialized")

        logger.success("Localizer initialization complete")

    def localize_frame(self, frame_rgb: np.ndarray) -> dict:
        """Process single frame and return GPS coordinates"""
        if not self.calibration.is_calibrated:
            logger.warning("Localization attempted without GPS calibration")
            return {"success": False, "error": "Система не відкалібрована"}

        height, width = frame_rgb.shape[:2]
        center_pt = np.array([[width / 2.0, height / 2.0]], dtype=np.float32)

        logger.debug(f"Processing frame: {width}x{height}")

        # Extract features from query frame
        logger.debug("Extracting query features...")
        query_features = self.feature_extractor.extract_features(frame_rgb)
        logger.debug(f"Extracted {len(query_features['keypoints'])} keypoints")

        # Find candidate frames
        top_k = self.config.get('localization', {}).get('retrieval_top_k', 5)
        logger.debug(f"Searching for top-{top_k} candidates...")
        candidates = self.retrieval.find_similar_frames(query_features['global_desc'], top_k=top_k)
        logger.debug(f"Found candidates: {[f'frame_{c[0]}' for c in candidates]}")

        best_inliers = 0
        best_homography = None
        best_candidate_id = -1

        # Try matching with each candidate
        for candidate_id, score in candidates:
            logger.debug(f"Matching with candidate {candidate_id} (similarity: {score:.4f})...")
            ref_features = self.database.get_local_features(candidate_id)

            mkpts_query, mkpts_ref = self.matcher.match(query_features, ref_features)
            logger.debug(f"Initial matches with candidate {candidate_id}: {len(mkpts_query)}")

            if len(mkpts_query) < self.min_matches:
                logger.debug(
                    f"Candidate {candidate_id}: insufficient matches ({len(mkpts_query)} < {self.min_matches})")
                continue

            H, mask = GeometryTransforms.estimate_homography(
                mkpts_query, mkpts_ref, ransac_threshold=self.ransac_thresh
            )

            if H is not None:
                inliers_count = int(np.sum(mask))
                logger.debug(f"Candidate {candidate_id}: {inliers_count} inliers after RANSAC")

                if inliers_count > best_inliers and inliers_count >= self.min_matches:
                    best_inliers = inliers_count
                    best_homography = H
                    best_candidate_id = candidate_id
                    logger.debug(f"New best match: frame {candidate_id} with {inliers_count} inliers")

        if best_homography is None:
            logger.warning("No valid homography found - insufficient matches")
            return {"success": False, "error": "Недостатньо точок для розрахунку гомографії"}

        logger.info(f"Best match: frame {best_candidate_id} with {best_inliers} inliers")

        # Трансформуємо центральну точку кадру у систему координат бази даних
        pt_2d = GeometryTransforms.apply_homography(center_pt, best_homography)[0]
        logger.debug(f"Center point transformed to 2D: ({pt_2d[0]:.2f}, {pt_2d[1]:.2f})")

        # Перевірка викиду у метричних координатах перед фільтрацією
        metric_pt = GeometryTransforms.apply_affine(
            np.array([pt_2d], dtype=np.float32),
            self.calibration.affine_matrix
        )[0]
        logger.debug(f"Metric coordinates (raw): ({metric_pt[0]:.2f}, {metric_pt[1]:.2f})")

        # ВИПРАВЛЕНО: перевіряємо викид до фільтрації
        if self.outlier_detector.is_outlier(metric_pt):
            logger.warning("Outlier detected - rejecting measurement")
            return {"success": False, "error": "Виявлено аномальний стрибок координат"}

        # ВИПРАВЛЕНО: застосовуємо Kalman filter і використовуємо його результат
        filtered_metric_pt = self.kalman_filter.update(metric_pt)
        self.outlier_detector.add_position(filtered_metric_pt)
        logger.debug(f"Filtered metric coordinates: ({filtered_metric_pt[0]:.2f}, {filtered_metric_pt[1]:.2f})")

        # ВИПРАВЛЕНО: конвертуємо у GPS з відфільтрованих метричних координат,
        # а не з сирого pt_2d як було раніше
        from src.geometry.coordinates import CoordinateConverter
        lat, lon = CoordinateConverter.metric_to_gps(filtered_metric_pt[0], filtered_metric_pt[1])
        logger.debug(f"GPS from filtered metric: ({lat:.6f}, {lon:.6f})")

        # Розрахунок поля зору (FOV) — трансформуємо 4 кути кадру
        corners = np.array([
            [0,     0],
            [width, 0],
            [width, height],
            [0,     height]
        ], dtype=np.float32)

        ref_corners_2d = GeometryTransforms.apply_homography(corners, best_homography)

        gps_corners = []
        for cx, cy in ref_corners_2d:
            try:
                corner_metric = GeometryTransforms.apply_affine(
                    np.array([[cx, cy]], dtype=np.float32),
                    self.calibration.affine_matrix
                )[0]
                c_lat, c_lon = CoordinateConverter.metric_to_gps(corner_metric[0], corner_metric[1])
                gps_corners.append((c_lat, c_lon))
            except Exception as e:
                logger.warning(f"Не вдалося трансформувати кут у GPS: {e}")

        confidence = min(1.0, best_inliers / 50.0)
        logger.success(f"Localization successful: ({lat:.6f}, {lon:.6f}), confidence: {confidence:.2f}")

        return {
            "success": True,
            "lat": lat,
            "lon": lon,
            "confidence": confidence,
            "matched_frame": best_candidate_id,
            "inliers": best_inliers,
            "fov_polygon": gps_corners
        }