import cv2
import numpy as np
from src.geometry.transformations import GeometryTransforms
from src.localization.matcher import FastRetrieval
from src.tracking.kalman_filter import TrajectoryFilter
from src.tracking.outlier_detector import OutlierDetector
from src.utils.logging_utils import get_logger
from src.geometry.coordinates import CoordinateConverter

logger = get_logger(__name__)

_ROTATIONS: list[tuple[int, int | None]] = [
    (0,   None),
    (90,  cv2.ROTATE_90_CLOCKWISE),
    (180, cv2.ROTATE_180),
    (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
]


class Localizer:
    """
    Drone localization pipeline з multi-anchor підтримкою.

    Ланцюжок координат (після пропагації):
    query → H(query→ref) → ref_space
          → H(ref→anchor) → anchor_space (pre-computed)
          → affine_anchor → metric
          → metric_to_gps → (lat, lon)
    """

    def __init__(self, database, feature_extractor, matcher, calibration, config=None):
        self.database = database
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.calibration = calibration
        self.config = config or {}

        loc_cfg = self.config.get('localization', {})
        trk_cfg = self.config.get('tracking', {})

        self.min_matches = loc_cfg.get('min_matches', 15)
        self.ransac_thresh = loc_cfg.get('ransac_threshold', 3.0)
        self.retrieval_top_k = loc_cfg.get('retrieval_top_k', 5)
        self.early_stop_inliers = loc_cfg.get('early_stop_inliers', 20)
        self.confidence_scale = loc_cfg.get('confidence_max_inliers', 50)

        self.retrieval = FastRetrieval(self.database.global_descriptors)
        self.kalman_filter = TrajectoryFilter(
            process_noise=trk_cfg.get('kalman_process_noise', 0.1),
            measurement_noise=trk_cfg.get('kalman_measurement_noise', 10.0),
        )
        self.outlier_detector = OutlierDetector(
            threshold_std=trk_cfg.get('outlier_threshold_std', 100.0),
            max_speed_mps=trk_cfg.get('max_speed_mps', 100000.0),
        )

        if self.database.is_propagated:
            n = int(np.sum(self.database.frame_valid))
            logger.success(f"Localizer: propagation data found ({n} frames pre-computed)")
        else:
            logger.warning("Localizer: no propagation data — run CalibrationPropagationWorker")

        logger.success("Localizer ready")

    def localize_frame(
        self,
        frame_rgb: np.ndarray,
        static_mask: np.ndarray | None = None,
        dt: float = 1.0,
    ) -> dict:
        if not self.calibration.is_calibrated:
            return {"success": False, "error": "Система не відкалібрована"}

        # 1. Global descriptor once — rotation-agnostic retrieval
        base_features = self.feature_extractor.extract_features(frame_rgb, static_mask=static_mask)
        candidates = self.retrieval.find_similar_frames(
            base_features['global_desc'], top_k=self.retrieval_top_k
        )

        best_inliers = 0
        best_H = None
        best_candidate_id = -1
        best_rot_angle = 0
        best_rot_code = None
        best_shape = frame_rgb.shape[:2]  # (height, width)

        # 2. Try each rotation; exit early on good match
        found = False
        for angle, rot_code in _ROTATIONS:
            if found:
                break

            if rot_code is not None:
                curr_rgb = cv2.rotate(frame_rgb, rot_code)
                curr_mask = cv2.rotate(static_mask, rot_code) if static_mask is not None else None
                # Only local features — skip DINOv2
                query_features = self.feature_extractor.extract_local_features(curr_rgb, curr_mask)
            else:
                query_features = base_features
                curr_rgb = frame_rgb

            for candidate_id, _score in candidates:
                ref_features = self.database.get_local_features(candidate_id)
                mkpts_q, mkpts_r = self.matcher.match(query_features, ref_features)

                if len(mkpts_q) < self.min_matches:
                    continue

                H, mask = GeometryTransforms.estimate_homography(
                    mkpts_q, mkpts_r, ransac_threshold=self.ransac_thresh
                )
                if H is None:
                    continue

                inliers = int(np.sum(mask))
                if inliers > best_inliers and inliers >= self.min_matches:
                    best_inliers = inliers
                    best_H = H
                    best_candidate_id = candidate_id
                    best_rot_angle = angle
                    best_rot_code = rot_code
                    best_shape = curr_rgb.shape[:2]

                if best_inliers >= self.early_stop_inliers:
                    logger.info(f"Early stop at {angle}° with {best_inliers} inliers")
                    found = True
                    break

        if best_H is None:
            return {"success": False, "error": "Не знайдено збігів у жодній орієнтації"}

        # 3. Retrieve affine for best candidate
        affine_ref = self.database.get_frame_affine(best_candidate_id)
        if affine_ref is None:
            return {"success": False, "error": "Немає GPS-даних. Запустіть пропагацію!"}

        # 4. Project image center through H → affine → GPS
        height, width = best_shape
        center_pt = np.array([[width / 2.0, height / 2.0]], dtype=np.float32)
        pt_in_ref = GeometryTransforms.apply_homography(center_pt, best_H)
        if pt_in_ref is None or len(pt_in_ref) == 0:
            return {"success": False, "error": "Помилка трансформації координат"}

        metric_pt = GeometryTransforms.apply_affine(pt_in_ref, affine_ref)[0]

        if self.outlier_detector.is_outlier(metric_pt, dt=dt):
            return {"success": False, "error": "Виявлено аномальний стрибок координат"}

        filtered_pt = self.kalman_filter.update_with_dt(metric_pt, dt)
        self.outlier_detector.add_position(filtered_pt)

        lat, lon = CoordinateConverter.metric_to_gps(filtered_pt[0], filtered_pt[1])

        # 5. FOV polygon
        corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        gps_corners = []
        ref_corners = GeometryTransforms.apply_homography(corners, best_H)
        if ref_corners is not None:
            metric_corners = GeometryTransforms.apply_affine(ref_corners, affine_ref)
            if metric_corners is not None:
                for cx, cy in metric_corners:
                    try:
                        gps_corners.append(CoordinateConverter.metric_to_gps(cx, cy))
                    except Exception as e:
                        logger.warning(f"FOV corner conversion failed: {e}")

        confidence = min(1.0, best_inliers / self.confidence_scale)
        logger.success(
            f"Localized: ({lat:.6f}, {lon:.6f}), frame={best_candidate_id}, "
            f"rot={best_rot_angle}°, inliers={best_inliers}"
        )
        return {
            "success": True,
            "lat": lat,
            "lon": lon,
            "confidence": confidence,
            "matched_frame": best_candidate_id,
            "inliers": best_inliers,
            "fov_polygon": gps_corners,
            "rotation_angle": best_rot_angle,
        }

    def reset_cache(self):
        self._runtime_h_cache = {}
        self._anchor_features_cache = {}
        logger.info("Localizer runtime cache cleared")
