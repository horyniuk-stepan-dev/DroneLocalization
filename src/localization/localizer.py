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

        global_descs = self.database.global_descriptors
        self.retrieval = FastRetrieval(global_descs)

        self.kalman_filter = TrajectoryFilter(
            process_noise=self.config.get('tracking', {}).get('kalman_process_noise', 0.1),
            measurement_noise=self.config.get('tracking', {}).get('kalman_measurement_noise', 10.0)
        )
        self.outlier_detector = OutlierDetector(
            threshold_std=self.config.get('tracking', {}).get('outlier_threshold_std', 100.0),
            max_speed_mps=self.config.get('tracking', {}).get('max_speed_mps', 100000.0)
        )

        # Runtime fallback кеш (якщо пропагація не виконана)
        self._runtime_h_cache: dict = {}
        self._anchor_features_cache: dict = {}

        if self.database.is_propagated:
            logger.success(
                f"Localizer init: propagation data found "
                f"({int(np.sum(self.database.frame_valid))} frames pre-computed)"
            )
        else:
            logger.warning(
                "Localizer init: no propagation data. "
                "Run CalibrationPropagationWorker for accurate multi-anchor results."
            )

        logger.success("Localizer ready")

    # ──────────────────────────────────────────────────────────────────
    # H(ref → anchor) — з БД або fallback
    # ──────────────────────────────────────────────────────────────────

    def _get_anchor_for_ref(self, candidate_id: int, ref_features: dict):
        """
        Повертає (H_ref_to_anchor, anchor) для candidate frame.
        H_ref_to_anchor — матриця що переводить координати з простору
        candidate frame у простір його найближчого якоря.

        Якщо candidate_id == anchor.frame_id → H = None (identity).
        """
        if not self.calibration.is_calibrated:
            return None, None

        # 1. Pre-computed з БД — O(1)
        db_result = self.database.get_h_to_anchor(candidate_id)
        if db_result is not None:
            H_to_anchor, anchor_fid = db_result
            anchor = next(
                (a for a in self.calibration.anchors if a.frame_id == anchor_fid), None
            )
            if anchor is not None:
                # identity означає що candidate_id сам є якорем
                if anchor_fid == candidate_id:
                    return None, anchor
                return H_to_anchor, anchor

        # 2. Runtime fallback — ланцюжок до найближчого якоря
        anchor = self._find_nearest_anchor(candidate_id)
        if anchor is None:
            return None, None

        if candidate_id == anchor.frame_id:
            return None, anchor

        cache_key = (candidate_id, anchor.frame_id)
        if cache_key in self._runtime_h_cache:
            return self._runtime_h_cache[cache_key], anchor

        logger.info(f"Runtime: computing H(frame_{candidate_id} → anchor_{anchor.frame_id})")
        anchor_feat = self._load_anchor_features(anchor.frame_id)
        if anchor_feat is None:
            return None, anchor

        H = self._build_chain(candidate_id, anchor.frame_id, ref_features, anchor_feat)
        self._runtime_h_cache[cache_key] = H
        return H, anchor

    def _find_nearest_anchor(self, frame_id: int):
        """Знайти найближчий якір за номером кадру"""
        if not self.calibration.anchors:
            return None
        return min(
            self.calibration.anchors,
            key=lambda a: abs(a.frame_id - frame_id)
        )

    def _load_anchor_features(self, anchor_fid: int) -> dict | None:
        if anchor_fid in self._anchor_features_cache:
            return self._anchor_features_cache[anchor_fid]
        try:
            feat = self.database.get_local_features(anchor_fid)
            self._anchor_features_cache[anchor_fid] = feat
            return feat
        except Exception as e:
            logger.error(f"Cannot load anchor {anchor_fid} features: {e}")
            return None

    def _match_pair(self, fa: dict, fb: dict) -> np.ndarray | None:
        try:
            mkpts_a, mkpts_b = self.matcher.match(fa, fb)
            if len(mkpts_a) < self.min_matches:
                return None
            H, mask = GeometryTransforms.estimate_homography(
                mkpts_a, mkpts_b, ransac_threshold=self.ransac_thresh
            )
            if H is None or int(np.sum(mask)) < self.min_matches:
                return None
            return H
        except Exception:
            return None

    def _build_chain(
            self, from_id: int, to_id: int,
            from_features: dict, to_features: dict,
            step: int = 10
    ) -> np.ndarray | None:
        """Ланцюжок H(from → to) через проміжні кадри"""
        H = self._match_pair(from_features, to_features)
        if H is not None:
            return H

        direction = 1 if to_id > from_id else -1
        waypoints = list(range(from_id, to_id, direction * step))
        if not waypoints or waypoints[-1] != to_id:
            waypoints.append(to_id)

        H_chain = []
        prev_feat = from_features

        for i in range(1, len(waypoints)):
            wp_id = waypoints[i]
            try:
                next_feat = (to_features if wp_id == to_id
                             else self.database.get_local_features(wp_id))
            except Exception:
                return None

            H_step = self._match_pair(prev_feat, next_feat)
            if H_step is None:
                return None

            H_chain.append(H_step)
            prev_feat = next_feat

        if not H_chain:
            return None

        H_result = np.eye(3, dtype=np.float64)
        for H in H_chain:
            H_result = H.astype(np.float64) @ H_result
        return H_result.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────
    # Трансформація точок
    # ──────────────────────────────────────────────────────────────────

    def _transform_pts(
            self,
            pts: np.ndarray,
            H_query_to_ref: np.ndarray,
            H_ref_to_anchor: np.ndarray | None
    ) -> np.ndarray | None:
        """
        pts (query space) → anchor space.
        H_ref_to_anchor=None означає ref і є якором.
        """
        pts_2d = pts.reshape(-1, 2)

        in_ref = GeometryTransforms.apply_homography(pts_2d, H_query_to_ref)
        if in_ref is None or len(in_ref) == 0:
            return None

        if H_ref_to_anchor is None:
            return in_ref

        in_anchor = GeometryTransforms.apply_homography(in_ref, H_ref_to_anchor)
        if in_anchor is None or len(in_anchor) == 0:
            return None

        return in_anchor

    # ──────────────────────────────────────────────────────────────────
    # Локалізація (З AUTO-ROTATION)
    # ──────────────────────────────────────────────────────────────────

    def localize_frame(self, frame_rgb: np.ndarray, static_mask: np.ndarray = None, dt: float = 1.0) -> dict:
        if not self.calibration.is_calibrated:
            return {"success": False, "error": "Система не відкалібрована"}

        # Список поворотів: (Кут, Код OpenCV)
        rotations = [
            (0, None),
            (90, cv2.ROTATE_90_CLOCKWISE),
            (180, cv2.ROTATE_180),
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ]

        best_inliers = 0
        best_H_query_to_ref = None
        best_candidate_id = -1
        best_rot_angle = 0
        best_rot_code = None

        top_k = self.config.get('localization', {}).get('retrieval_top_k', 5)

        # 1. Шукаємо збіги, перебираючи кути повороту
        for angle, rot_code in rotations:
            # Повертаємо зображення та маску
            if rot_code is not None:
                curr_rgb = cv2.rotate(frame_rgb, rot_code)
                curr_mask = cv2.rotate(static_mask, rot_code) if static_mask is not None else None
            else:
                curr_rgb = frame_rgb
                curr_mask = static_mask

            # Витягуємо фічі з поточного повороту
            query_features = self.feature_extractor.extract_features(curr_rgb, static_mask=curr_mask)
            candidates = self.retrieval.find_similar_frames(query_features['global_desc'], top_k=top_k)

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
                        best_rot_angle = angle
                        best_rot_code = rot_code

            # Якщо знайшли хороший збіг — зупиняємо перебір, економимо час
            if best_inliers >= 20:
                logger.info(f"Early stop on rotation {angle}° with {best_inliers} inliers")
                break

        if best_H_query_to_ref is None:
            return {"success": False, "error": "Не знайдено збігів у жодній орієнтації (0°, 90°, 180°, 270°)"}

        # 2. Отримуємо матрицю знайденого кадру з бази
        affine_ref = self.database.get_frame_affine(best_candidate_id)
        if affine_ref is None:
            return {"success": False, "error": "Немає даних GPS для цього кадру. Запустіть пропагацію!"}

        # 3. Рахуємо розміри ПОВЕРНУТОГО зображення (адже знайдена матриця H відповідає саме йому)
        if best_rot_code is not None:
            final_rgb = cv2.rotate(frame_rgb, best_rot_code)
        else:
            final_rgb = frame_rgb

        height, width = final_rgb.shape[:2]
        center_pt = np.array([[width / 2.0, height / 2.0]], dtype=np.float32)

        # 4. Трансформуємо центр у систему знайденого кадру
        pt_in_ref = GeometryTransforms.apply_homography(center_pt, best_H_query_to_ref)
        if pt_in_ref is None or len(pt_in_ref) == 0:
            return {"success": False, "error": "Помилка трансформації координат"}

        # 5. Переводимо в метрику і перевіряємо на аномалії (враховуючи dt)
        metric_pt = GeometryTransforms.apply_affine(pt_in_ref, affine_ref)[0]

        if self.outlier_detector.is_outlier(metric_pt, dt=dt):
            return {"success": False, "error": "Виявлено аномальний стрибок координат"}

        # Оновлення Калмана з dt
        if hasattr(self.kalman_filter, 'update_with_dt'):
            filtered_pt = self.kalman_filter.update_with_dt(metric_pt, dt)
        else:
            filtered_pt = self.kalman_filter.update(metric_pt)

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
            "fov_polygon": gps_corners,
            "rotation_angle": best_rot_angle
        }

    def reset_cache(self):
        logger.info("Localizer runtime cache cleared")