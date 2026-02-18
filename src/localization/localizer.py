import numpy as np
from src.geometry.transformations import GeometryTransforms
from src.geometry.coordinates import CoordinateConverter
from src.localization.matcher import FastRetrieval
from src.tracking.kalman_filter import TrajectoryFilter
from src.tracking.outlier_detector import OutlierDetector
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Localizer:
    """
    Main drone localization pipeline.

    Повний ланцюжок координат:
        query → H(query→ref) → ref_space
             → H(ref→calib)  → calib_space   (з БД після пропагації)
             → affine_matrix → metric
             → metric_to_gps → (lat, lon)

    H(ref→calib) читається з бази даних (pre-computed CalibrationPropagationWorker).
    Якщо пропагація ще не виконана — будує ланцюжок на льоту (fallback).
    """

    def __init__(self, database, feature_extractor, matcher, calibration, config=None):
        self.database = database
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.calibration = calibration
        self.config = config or {}

        self.min_matches = self.config.get('localization', {}).get('min_matches', 15)
        self.ransac_thresh = self.config.get('localization', {}).get('ransac_threshold', 3.0)
        self.chain_step = self.config.get('localization', {}).get('chain_step', 10)

        logger.info("Initializing Localizer...")

        global_descs = self.database.global_descriptors
        self.retrieval = FastRetrieval(global_descs)

        self.kalman_filter = TrajectoryFilter(
            process_noise=self.config.get('tracking', {}).get('kalman_process_noise', 0.1),
            measurement_noise=self.config.get('tracking', {}).get('kalman_measurement_noise', 10.0)
        )
        self.outlier_detector = OutlierDetector(
            threshold_std=self.config.get('tracking', {}).get('outlier_threshold_std', 3.0)
        )

        # Fallback кеш для випадку коли пропагація не виконана
        self._runtime_h_cache: dict = {}
        self._calib_features = None
        self._load_calib_features()

        if self.database.is_propagated:
            logger.success("Propagation data found — using pre-computed H_to_calib (fast mode)")
        else:
            logger.warning(
                "No propagation data found. "
                "Run CalibrationPropagationWorker after calibration for best accuracy. "
                "Falling back to runtime chain computation."
            )

        logger.success("Localizer initialization complete")

    # ------------------------------------------------------------------
    # Ініціалізація
    # ------------------------------------------------------------------

    def _load_calib_features(self):
        if not self.calibration.is_calibrated:
            return
        calib_id = self.calibration.calib_frame_id
        try:
            self._calib_features = self.database.get_local_features(calib_id)
            logger.info(f"Loaded calib_frame {calib_id}: "
                        f"{len(self._calib_features['keypoints'])} kpts")
        except Exception as e:
            logger.error(f"Failed to load calib_frame {calib_id}: {e}")
            self._calib_features = None

    # ------------------------------------------------------------------
    # Отримання H(ref → calib_frame)
    # ------------------------------------------------------------------

    def _get_h_ref_to_calib(self, candidate_id: int, ref_features: dict) -> np.ndarray | None:
        """
        Пріоритет:
        1. БД (pre-computed) — O(1), завжди точно.
        2. Runtime кеш — якщо вже рахували в цій сесії.
        3. Ланцюжок на льоту — тільки якщо пропагація не виконана.
        """
        calib_id = self.calibration.calib_frame_id

        if candidate_id == calib_id:
            return None  # identity, крок не потрібен

        # 1. Pre-computed з БД
        H_db = self.database.get_h_to_calib(candidate_id)
        if H_db is not None:
            logger.debug(f"H(frame_{candidate_id}→calib): from DB (pre-computed)")
            return H_db

        # 2. Runtime кеш
        if candidate_id in self._runtime_h_cache:
            logger.debug(f"H(frame_{candidate_id}→calib): from runtime cache")
            return self._runtime_h_cache[candidate_id]

        # 3. Fallback: будуємо ланцюжок на льоту
        logger.info(f"H(frame_{candidate_id}→calib_{calib_id}): computing chain (fallback)...")
        H = self._build_chain(candidate_id, calib_id, ref_features)
        self._runtime_h_cache[candidate_id] = H
        return H

    def _match_pair(self, features_a: dict, features_b: dict) -> np.ndarray | None:
        """Зіставити два кадри → H(a→b) або None"""
        mkpts_a, mkpts_b = self.matcher.match(features_a, features_b)
        if len(mkpts_a) < self.min_matches:
            return None
        H, mask = GeometryTransforms.estimate_homography(
            mkpts_a, mkpts_b, ransac_threshold=self.ransac_thresh
        )
        if H is None or int(np.sum(mask)) < self.min_matches:
            return None
        return H

    def _build_chain(
        self, from_id: int, to_id: int, from_features: dict
    ) -> np.ndarray | None:
        """
        Будує H(from_id → to_id) через проміжні кадри з кроком chain_step.
        """
        if self._calib_features is None:
            return None

        # Прямий матчинг
        H = self._match_pair(from_features, self._calib_features)
        if H is not None:
            return H

        # Ланцюжок
        step = self.chain_step if to_id > from_id else -self.chain_step
        waypoints = list(range(from_id, to_id, step))
        if not waypoints or waypoints[-1] != to_id:
            waypoints.append(to_id)

        H_chain = []
        prev_features = from_features

        for i in range(1, len(waypoints)):
            wp_id = waypoints[i]
            next_feat = (self._calib_features if wp_id == to_id
                         else self._try_load(wp_id))
            if next_feat is None:
                return None

            H_step = self._match_pair(prev_features, next_feat)
            if H_step is None:
                # Спробуємо midpoint
                mid_id = (waypoints[i - 1] + wp_id) // 2
                if mid_id not in (waypoints[i - 1], wp_id):
                    mid_feat = self._try_load(mid_id)
                    if mid_feat is not None:
                        H1 = self._match_pair(prev_features, mid_feat)
                        H2 = self._match_pair(mid_feat, next_feat)
                        if H1 is not None and H2 is not None:
                            H_step = (H2.astype(np.float64) @ H1.astype(np.float64)).astype(np.float32)
                if H_step is None:
                    logger.warning(f"Chain broken at step {waypoints[i-1]}→{wp_id}")
                    return None

            H_chain.append(H_step)
            prev_features = next_feat

        if not H_chain:
            return None

        H_result = np.eye(3, dtype=np.float64)
        for H in H_chain:
            H_result = H.astype(np.float64) @ H_result
        return H_result.astype(np.float32)

    def _try_load(self, frame_id: int) -> dict | None:
        try:
            return self.database.get_local_features(frame_id)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Трансформація точок
    # ------------------------------------------------------------------

    def _to_calib_space(
        self,
        pts: np.ndarray,
        H_query_to_ref: np.ndarray,
        H_ref_to_calib: np.ndarray | None
    ) -> np.ndarray | None:
        """
        pts (query space) → calib_frame space.
        Якщо H_ref_to_calib is None — best_match і є calib_frame.
        """
        pts_2d = pts.reshape(-1, 2)

        in_ref = GeometryTransforms.apply_homography(pts_2d, H_query_to_ref)
        if in_ref is None or len(in_ref) == 0:
            return None

        if H_ref_to_calib is None:
            return in_ref

        in_calib = GeometryTransforms.apply_homography(in_ref, H_ref_to_calib)
        if in_calib is None or len(in_calib) == 0:
            return None

        return in_calib

    # ------------------------------------------------------------------
    # Основний метод
    # ------------------------------------------------------------------

    def localize_frame(self, frame_rgb: np.ndarray) -> dict:
        """Process single frame and return GPS coordinates"""
        if not self.calibration.is_calibrated:
            return {"success": False, "error": "Система не відкалібрована"}

        if not self.database.is_propagated:
            logger.warning(
                "GPS propagation not done yet! "
                "Run 'Calibration → Propagate GPS to all frames' for accurate results."
            )

        if self._calib_features is None:
            self._load_calib_features()
            if self._calib_features is None:
                return {"success": False,
                        "error": "Не вдалося завантажити кадр калібрування з бази"}

        height, width = frame_rgb.shape[:2]
        center_pt = np.array([[width / 2.0, height / 2.0]], dtype=np.float32)

        # Витягуємо фічі
        query_features = self.feature_extractor.extract_features(frame_rgb)

        # Знаходимо кандидатів
        top_k = self.config.get('localization', {}).get('retrieval_top_k', 5)
        candidates = self.retrieval.find_similar_frames(query_features['global_desc'], top_k=top_k)

        best_inliers = 0
        best_H_query_to_ref = None
        best_candidate_id = -1
        best_ref_features = None

        for candidate_id, score in candidates:
            ref_features = self.database.get_local_features(candidate_id)
            mkpts_q, mkpts_r = self.matcher.match(query_features, ref_features)

            if len(mkpts_q) < self.min_matches:
                continue

            H, mask = GeometryTransforms.estimate_homography(
                mkpts_q, mkpts_r, ransac_threshold=self.ransac_thresh
            )
            if H is not None:
                inliers_count = int(np.sum(mask))
                if inliers_count > best_inliers and inliers_count >= self.min_matches:
                    best_inliers = inliers_count
                    best_H_query_to_ref = H
                    best_candidate_id = candidate_id
                    best_ref_features = ref_features

        if best_H_query_to_ref is None:
            return {"success": False, "error": "Недостатньо точок для розрахунку гомографії"}

        logger.info(f"Best match: frame {best_candidate_id}, {best_inliers} inliers")

        # H(best_match → calib_frame) — з БД або fallback
        H_ref_to_calib = self._get_h_ref_to_calib(best_candidate_id, best_ref_features)

        if H_ref_to_calib is None and best_candidate_id != self.calibration.calib_frame_id:
            logger.warning("Could not get H_ref_to_calib, using direct affine (reduced accuracy)")
            pts_in_calib = GeometryTransforms.apply_homography(center_pt, best_H_query_to_ref)
        else:
            pts_in_calib = self._to_calib_space(center_pt, best_H_query_to_ref, H_ref_to_calib)

        if pts_in_calib is None or len(pts_in_calib) == 0:
            return {"success": False, "error": "Помилка трансформації координат"}

        pt_in_calib = pts_in_calib[0]
        logger.debug(f"Center in calib_frame: ({pt_in_calib[0]:.2f}, {pt_in_calib[1]:.2f})")

        # calib pixel → metric → GPS
        try:
            metric_x, metric_y = self.calibration.pixel_to_metric(pt_in_calib[0], pt_in_calib[1])
        except Exception as e:
            return {"success": False, "error": f"Помилка affine: {e}"}

        metric_np = np.array([metric_x, metric_y], dtype=np.float32)

        if self.outlier_detector.is_outlier(metric_np):
            return {"success": False, "error": "Виявлено аномальний стрибок координат"}

        filtered = self.kalman_filter.update(metric_np)
        self.outlier_detector.add_position(filtered)

        lat, lon = CoordinateConverter.metric_to_gps(filtered[0], filtered[1])

        # FOV
        corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        gps_corners = []

        if H_ref_to_calib is None and best_candidate_id != self.calibration.calib_frame_id:
            corners_in_calib = GeometryTransforms.apply_homography(corners, best_H_query_to_ref)
        else:
            corners_in_calib = self._to_calib_space(corners, best_H_query_to_ref, H_ref_to_calib)

        if corners_in_calib is not None:
            for cx, cy in corners_in_calib:
                try:
                    mx, my = self.calibration.pixel_to_metric(cx, cy)
                    c_lat, c_lon = CoordinateConverter.metric_to_gps(mx, my)
                    gps_corners.append((c_lat, c_lon))
                except Exception:
                    pass

        confidence = min(1.0, best_inliers / 50.0)
        logger.success(f"Localization: ({lat:.6f}, {lon:.6f}), confidence={confidence:.2f}")

        return {
            "success": True,
            "lat": lat,
            "lon": lon,
            "confidence": confidence,
            "matched_frame": best_candidate_id,
            "inliers": best_inliers,
            "fov_polygon": gps_corners
        }

    def reset_cache(self):
        """Очистити runtime кеш (після перекалібрування)"""
        self._runtime_h_cache.clear()
        self._calib_features = None
        self._load_calib_features()
        logger.info("Localizer runtime cache cleared")