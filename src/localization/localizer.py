import cv2
import numpy as np
from src.geometry.transformations import GeometryTransforms
from src.localization.matcher import FastRetrieval
from src.tracking.kalman_filter import TrajectoryFilter
from src.tracking.outlier_detector import OutlierDetector
from src.utils.logging_utils import get_logger
from src.geometry.coordinates import CoordinateConverter

logger = get_logger(__name__)


class Localizer:
    """
    Drone localization pipeline з multi-anchor підтримкою.

    Ланцюжок координат (після пропагації):
        query → H(query→ref)  → ref_space
              → H(ref→anchor) → anchor_space  (з БД, pre-computed)
              → affine_anchor → metric
              → metric_to_gps → (lat, lon)

    H(ref→anchor) читається з БД (pre-computed CalibrationPropagationWorker).
    Якщо пропагація ще не виконана — fallback до runtime chain.
    """

    def __init__(self, database, feature_extractor, matcher, calibration, config=None):
        self.database = database
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.calibration = calibration  # MultiAnchorCalibration
        self.config = config or {}

        self.min_matches = self.config.get('localization', {}).get('min_matches', 15)
        self.ransac_thresh = self.config.get('localization', {}).get('ransac_threshold', 3.0)
        self.enable_auto_rotation = self.config.get('localization', {}).get('auto_rotation', True)

        self.trajectory_filter = TrajectoryFilter(
            process_noise=self.config.get('tracking', {}).get('process_noise', 0.1),
            measurement_noise=self.config.get('tracking', {}).get('measurement_noise', 10.0),
            dt=1.0  # Буде оновлюватись динамічно
        )
        self.outlier_detector = OutlierDetector(
            window_size=self.config.get('tracking', {}).get('outlier_window', 10),
            threshold_std=self.config.get('tracking', {}).get('outlier_std', 3.0),
            max_speed_mps=self.config.get('tracking', {}).get('max_speed_mps', 50.0)
        )

        logger.info("Initializing Localizer with DINOv2 + XFeat pipeline")

    def localize_frame(self, query_frame: np.ndarray, static_mask: np.ndarray = None, dt: float = 1.0) -> dict:
        """
        Локалізація поточного кадру (query).
        Повертає GPS координати центру та кути FOV.
        """
        logger.debug("Starting frame localization...")
        height, width = query_frame.shape[:2]

        # 1. Швидкий пошук кандидатів через глобальний дескриптор DINOv2
        query_features = self.feature_extractor.extract_features(query_frame, static_mask)
        retriever = FastRetrieval(self.database.global_descriptors)

        # Беремо top_k кандидатів
        top_k = self.config.get('localization', {}).get('top_k_candidates', 10)
        candidates = retriever.find_similar_frames(query_features['global_desc'], top_k=top_k)

        if not candidates:
            return {"success": False, "error": "No candidates found via DINOv2"}

        best_inliers = 0
        best_candidate_id = -1
        best_H_query_to_ref = None
        best_rot_angle = 0

        angles_to_try = [0]
        if self.enable_auto_rotation:
            angles_to_try = [0, 90, 180, 270]

        # 2. Перебір кандидатів і можливих обертань для метчингу XFeat
        for candidate_id, score in candidates:
            ref_features = self.database.get_local_features(candidate_id)

            for angle in angles_to_try:
                if angle == 0:
                    rot_features = query_features
                else:
                    k = angle // 90
                    rotated_frame = np.rot90(query_frame, k=k)
                    rotated_mask = np.rot90(static_mask, k=k) if static_mask is not None else None
                    rot_features = self.feature_extractor.extract_features(rotated_frame, static_mask=rotated_mask)

                mkpts_q, mkpts_r = self.matcher.match(rot_features, ref_features)

                if len(mkpts_q) >= self.min_matches:
                    H, mask = GeometryTransforms.estimate_homography(
                        mkpts_q, mkpts_r, ransac_threshold=self.ransac_thresh
                    )

                    if H is not None:
                        inliers = int(np.sum(mask))
                        if inliers > best_inliers:
                            best_inliers = inliers
                            best_candidate_id = candidate_id
                            best_rot_angle = angle

                            if angle != 0:
                                # Компенсація обертання матриці
                                H_rot = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
                                H_rot_3x3 = np.eye(3)
                                H_rot_3x3[:2, :] = H_rot
                                best_H_query_to_ref = H @ H_rot_3x3
                            else:
                                best_H_query_to_ref = H

        if best_inliers < self.min_matches or best_H_query_to_ref is None:
            return {"success": False, "error": f"Not enough XFeat inliers ({best_inliers} < {self.min_matches})"}

        # 3. Трансформація координат: query -> ref
        center_query = np.array([[width / 2.0, height / 2.0]], dtype=np.float32)
        center_ref = GeometryTransforms.apply_homography(center_query, best_H_query_to_ref)

        if center_ref is None:
            return {"success": False, "error": "Homography projection failed"}

        # 4. Перехід до метричних координат через пропаговану афінну матрицю
        affine_ref = self.database.get_frame_affine(best_candidate_id)
        if affine_ref is None:
            return {"success": False, "error": f"Frame {best_candidate_id} has no valid propagated calibration"}

        metric_pt = GeometryTransforms.apply_affine(center_ref, affine_ref)
        if metric_pt is None or len(metric_pt) == 0:
            return {"success": False, "error": "Affine projection to metric space failed"}

        metric_pt = metric_pt[0]

        # 5. Фільтрація траєкторії та відхилення аномалій
        if self.outlier_detector.is_outlier((metric_pt[0], metric_pt[1]), dt=dt):
            return {"success": False, "error": "Anomaly detected (velocity or Z-score)"}

        filtered_pt = self.trajectory_filter.update(metric_pt, dt=dt)
        self.outlier_detector.add_position(filtered_pt)
        lat, lon = CoordinateConverter.metric_to_gps(filtered_pt[0], filtered_pt[1])

        # 6. Розрахунок поля зору (FOV) для ПОВЕРНУТОГО кадру
        corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        ref_corners = GeometryTransforms.apply_homography(corners, best_H_query_to_ref)

        gps_corners = []
        if ref_corners is not None:
            metric_corners = GeometryTransforms.apply_affine(ref_corners, affine_ref)
            if metric_corners is not None:
                for cx, cy in metric_corners:
                    try:
                        clat, clon = CoordinateConverter.metric_to_gps(cx, cy)
                        gps_corners.append((clat, clon))
                    except Exception:
                        pass

        confidence = min(1.0, best_inliers / 50.0)
        logger.success(
            f"Localization: ({lat:.6f}, {lon:.6f}), matched frame={best_candidate_id}, rot={best_rot_angle}°")

        return {
            "success": True,
            "lat": lat,
            "lon": lon,
            "confidence": confidence,
            "matched_frame": best_candidate_id,
            "inliers": best_inliers,
            "fov_polygon": gps_corners
        }