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
    def __init__(self, database, feature_extractor, matcher, calibration, config=None):
        self.database = database
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.calibration = calibration
        self.config = config or {}

        self.min_matches = self.config.get('localization', {}).get('min_matches', 15)
        self.ransac_thresh = self.config.get('localization', {}).get('ransac_threshold', 3.0)
        self.enable_auto_rotation = self.config.get('localization', {}).get('auto_rotation', True)

        self.trajectory_filter = TrajectoryFilter(
            process_noise=self.config.get('tracking', {}).get('kalman_process_noise', 2.0),
            measurement_noise=self.config.get('tracking', {}).get('kalman_measurement_noise', 5.0),
            dt=1.0
        )
        self.outlier_detector = OutlierDetector(
            window_size=self.config.get('tracking', {}).get('outlier_window', 10),
            threshold_std=self.config.get('tracking', {}).get('outlier_threshold_std', 3.0),
            max_speed_mps=self.config.get('tracking', {}).get('max_speed_mps', 1000.0)
        )

        # Створюємо FastRetrieval один раз — нормалізація дескрипторів відбувається лише тут
        self.retriever = FastRetrieval(self.database.global_descriptors)

    def localize_frame(self, query_frame: np.ndarray, static_mask: np.ndarray = None, dt: float = 1.0) -> dict:
        height, width = query_frame.shape[:2]

        # 1. Екстракція для 0° та пошук кандидатів (ТІЛЬКИ 1 РАЗ)
        query_features = self.feature_extractor.extract_features(query_frame, static_mask)

        top_k = self.config.get('localization', {}).get('retrieval_top_k', 5)
        candidates = self.retriever.find_similar_frames(query_features['global_desc'], top_k=top_k)

        if not candidates:
            return {"success": False, "error": "No candidates found via DINOv2"}

        best_inliers = 0
        best_candidate_id = -1
        best_mkpts_q_inliers = None
        best_mkpts_r_inliers = None
        best_rot_angle = 0

        angles_to_try = [0, 90, 180, 270] if self.enable_auto_rotation else [0]
        features_cache = {0: query_features}  # КЕШ ОЗНАК ПОВЕРНЕННЯ
        early_stop = self.config.get('localization', {}).get('early_stop_inliers', 50)

        # 2. Перебір кандидатів
        for candidate_id, score in candidates:
            ref_features = self.database.get_local_features(candidate_id)

            for angle in angles_to_try:
                if angle not in features_cache:
                    k = angle // 90
                    rotated_frame = np.rot90(query_frame, k=k).copy()
                    rotated_mask = np.rot90(static_mask, k=k).copy() if static_mask is not None else None
                    features_cache[angle] = self.feature_extractor.extract_features(rotated_frame,
                                                                                    static_mask=rotated_mask)

                rot_features = features_cache[angle]
                mkpts_q, mkpts_r = self.matcher.match(rot_features, ref_features)

                if len(mkpts_q) >= self.min_matches:
                    H, mask = GeometryTransforms.estimate_homography(
                        mkpts_q, mkpts_r, ransac_threshold=self.ransac_thresh
                    )

                    if H is not None:
                        inlier_mask = mask.ravel().astype(bool)
                        inliers = int(np.sum(inlier_mask))
                        if inliers > best_inliers:
                            best_inliers = inliers
                            best_candidate_id = candidate_id
                            best_rot_angle = angle
                            best_mkpts_q_inliers = mkpts_q[inlier_mask]
                            best_mkpts_r_inliers = mkpts_r[inlier_mask]

                if best_inliers >= early_stop:
                    break
            if best_inliers >= early_stop:
                break

        if best_inliers < self.min_matches or best_mkpts_r_inliers is None:
            return {"success": False, "error": f"Not enough valid inliers ({best_inliers} < {self.min_matches})"}

        # Фізичні розміри ПОВЕРНУТОГО кадру
        if best_rot_angle in [90, 270]:
            rot_height, rot_width = width, height
        else:
            rot_height, rot_width = height, width

        # 3. Трансформація координат — через центроїд інлієрів (завжди в межах кадру!)
        centroid_ref = np.mean(best_mkpts_r_inliers, axis=0)  # Центр збігу в ref (в межах кадру)
        centroid_query = np.mean(best_mkpts_q_inliers, axis=0)

        # Зсув від центроїда збігу до центру кадру (в пікселях запиту)
        center_query = np.array([rot_width / 2.0, rot_height / 2.0], dtype=np.float32)
        offset_px = center_query - centroid_query  # Зсув в пікселях

        affine_ref = self.database.get_frame_affine(best_candidate_id)
        if affine_ref is None:
            return {"success": False, "error": "No propagated calibration"}

        # Центроїд ref → metric
        centroid_ref_arr = np.array([centroid_ref], dtype=np.float32)
        metric_centroid = GeometryTransforms.apply_affine(centroid_ref_arr, affine_ref)
        if metric_centroid is None or len(metric_centroid) == 0:
            return {"success": False, "error": "Projection failed"}
        metric_centroid = metric_centroid[0]

        # Перетворити піксельний зсув в метричний (через масштаб affine)
        scale_x = np.linalg.norm(affine_ref[0, :2])  # м/піксель по X
        scale_y = np.linalg.norm(affine_ref[1, :2])  # м/піксель по Y
        metric_offset = np.array([offset_px[0] * scale_x, offset_px[1] * scale_y], dtype=np.float32)

        metric_pt = metric_centroid + metric_offset

        # DEBUG: показати кожен крок
        logger.info(f"COORD DEBUG: centroid_ref=({centroid_ref[0]:.1f}, {centroid_ref[1]:.1f}), "
                     f"offset_px=({offset_px[0]:.1f}, {offset_px[1]:.1f}), frame={best_candidate_id}")
        logger.info(f"COORD DEBUG: metric_pt=({metric_pt[0]:.2f}, {metric_pt[1]:.2f})")

        # Перевіряємо чи нова точка — аномалія (стрибок координат)
        if self.outlier_detector.is_outlier(metric_pt, dt):
            return {"success": False, "error": "Outlier detected — position jump filtered"}

        # 4. Фільтр Калмана
        if hasattr(self.trajectory_filter, 'update_with_dt'):
            filtered_pt = self.trajectory_filter.update_with_dt(metric_pt, dt)
        else:
            filtered_pt = self.trajectory_filter.update(metric_pt, dt=dt)

        self.outlier_detector.add_position(filtered_pt)
        lat, lon = CoordinateConverter.metric_to_gps(filtered_pt[0], filtered_pt[1])

        # 5. Розрахунок FOV навколо метричної точки
        half_w = (rot_width / 2.0) * scale_x
        half_h = (rot_height / 2.0) * scale_y

        gps_corners = []
        fov_metric_corners = [
            (filtered_pt[0] - half_w, filtered_pt[1] - half_h),
            (filtered_pt[0] + half_w, filtered_pt[1] - half_h),
            (filtered_pt[0] + half_w, filtered_pt[1] + half_h),
            (filtered_pt[0] - half_w, filtered_pt[1] + half_h),
        ]
        for cx, cy in fov_metric_corners:
            try:
                clat, clon = CoordinateConverter.metric_to_gps(cx, cy)
                gps_corners.append((clat, clon))
            except Exception:
                pass

        confidence = min(1.0, best_inliers / 50.0)
        logger.success(
            f"Localization: ({lat:.6f}, {lon:.6f}), matched frame={best_candidate_id}, rot={best_rot_angle}°")

        return {
            "success": True, "lat": lat, "lon": lon,
            "confidence": confidence, "matched_frame": best_candidate_id,
            "inliers": best_inliers, "fov_polygon": gps_corners
        }