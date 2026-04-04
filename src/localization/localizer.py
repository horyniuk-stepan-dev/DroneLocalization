import numpy as np

from config.config import get_cfg
from src.geometry.transformations import GeometryTransforms
from src.localization.matcher import FastRetrieval
from src.tracking.kalman_filter import TrajectoryFilter
from src.tracking.outlier_detector import OutlierDetector
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Localizer:
    def __init__(self, database, feature_extractor, matcher, calibration, config=None):
        self.database = database
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.calibration = calibration
        self.config = config or {}

        # Дефолти синхронізовані з APP_CONFIG через get_cfg()
        self.min_matches = get_cfg(self.config, "localization.min_matches", 12)
        self.ransac_thresh = get_cfg(self.config, "localization.ransac_threshold", 3.0)
        self.enable_auto_rotation = get_cfg(self.config, "localization.auto_rotation", True)

        self.trajectory_filter = TrajectoryFilter(
            process_noise=get_cfg(self.config, "tracking.kalman_process_noise", 2.0),
            measurement_noise=get_cfg(self.config, "tracking.kalman_measurement_noise", 5.0),
            dt=1.0,
        )
        self.outlier_detector = OutlierDetector(
            window_size=get_cfg(self.config, "tracking.outlier_window", 10),
            threshold_std=get_cfg(self.config, "tracking.outlier_threshold_std", 25.0),
            max_speed_mps=get_cfg(self.config, "tracking.max_speed_mps", 1000.0),
            max_consecutive=get_cfg(self.config, "tracking.max_consecutive_outliers", 5),
        )

        # Створюємо FastRetrieval один раз — нормалізація дескрипторів відбувається лише тут
        self.retriever = FastRetrieval(self.database.global_descriptors)

        # Fallback: SuperPoint+LightGlue для складних сцен
        self.model_manager = self.config.get("_model_manager", None)
        self.fallback_enabled = get_cfg(self.config, "localization.enable_lightglue_fallback", True)
        self.min_inliers_for_accept = get_cfg(self.config, "localization.min_inliers_accept", 10)
        self.retrieval_top_k = get_cfg(self.config, "localization.retrieval_top_k", 8)
        self.early_stop_inliers = get_cfg(self.config, "localization.early_stop_inliers", 30)

        # Fix #1: Захист від нескінченного циклу при виході за межі покриття
        self._consecutive_failures = 0
        self._max_failures = get_cfg(self.config, "localization.max_consecutive_failures", 10)

    def localize_frame(
        self, query_frame: np.ndarray, static_mask: np.ndarray = None, dt: float = 1.0
    ) -> dict:
        # Fix #1: Якщо було занадто багато послідовних невдач — повертаємо out_of_coverage
        if self._consecutive_failures >= self._max_failures:
            self._consecutive_failures = 0
            logger.warning(
                f"Out-of-coverage guard triggered after {self._max_failures} consecutive failures. "
                f"Resetting counter. The drone may be outside the database coverage area."
            )
            return {
                "success": False,
                "error": "out_of_coverage",
                "detail": f"Exceeded {self._max_failures} consecutive localization failures",
            }

        height, width = query_frame.shape[:2]

        angles_to_try = [0, 90, 180, 270] if self.enable_auto_rotation else [0]

        best_global_score = -1.0
        best_global_angle = 0
        best_global_candidates = []
        best_query_features = None

        top_k = self.retrieval_top_k

        # 1. Екстракція ознак для всіх дозволених кутів обертання та вибір найкращого ракурсу
        for angle in angles_to_try:
            k = angle // 90
            rotated_frame = np.rot90(query_frame, k=k).copy()

            # Витягуємо ТІЛЬКИ глобальний дескриптор DINOv2 для швидкого пошуку ракурсу
            global_desc = self.feature_extractor.extract_global_descriptor(rotated_frame)

            # Шукаємо кандидатів за допомогою DINOv2
            candidates = self.retriever.find_similar_frames(global_desc, top_k=top_k)

            if candidates:
                # Оцінкою ракурсу вважаємо скор найкращого кандидата
                top_score = candidates[0][1]
                if top_score > best_global_score:
                    best_global_score = top_score
                    best_global_angle = angle
                    best_global_candidates = candidates

        if not best_global_candidates:
            self._consecutive_failures += 1
            return {
                "success": False,
                "error": (
                    f"No candidates found via global descriptor (DINOv2) in any rotation. "
                    f"Tested angles: {angles_to_try}. "
                    f"Image {width}x{height} may not match any frame in the database."
                ),
            }

        logger.info(
            f"Selected best rotation {best_global_angle}° with global score {best_global_score:.3f}"
        )

        # 1.5. Локальна екстракція (XFeat) ТІЛЬКИ для НАЙКРАЩОГО ракурсу
        k = best_global_angle // 90
        best_rotated_frame = np.rot90(query_frame, k=k).copy()
        best_rotated_mask = np.rot90(static_mask, k=k).copy() if static_mask is not None else None

        # Обчислюємо ключові точки лише один раз для обраного кута!
        best_query_features = self.feature_extractor.extract_local_features(
            best_rotated_frame, static_mask=best_rotated_mask
        )

        best_inliers = 0
        best_candidate_id = -1
        best_H_query_to_ref = None
        best_mkpts_q_inliers = None
        best_mkpts_r_inliers = None
        best_total_matches = 0
        best_rmse = 999.0

        early_stop = self.early_stop_inliers

        # 2. Локальний пошук (XFeat) ТІЛЬКИ для найкращого знайденого ракурсу
        for candidate_id, score in best_global_candidates:
            logger.debug(f"  → Trying candidate {candidate_id} (global_score={score:.3f})")
            ref_features = self.database.get_local_features(candidate_id)

            mkpts_q, mkpts_r = self.matcher.match(best_query_features, ref_features)

            if len(mkpts_q) >= self.min_matches:
                # Використовуємо Homography (8 DoF)
                H_eval, mask = GeometryTransforms.estimate_homography(
                    mkpts_q, mkpts_r, ransac_threshold=self.ransac_thresh
                )

                if H_eval is not None:
                    inlier_mask = mask.ravel().astype(bool)
                    inliers = int(np.sum(inlier_mask))
                    pts_q_in = mkpts_q[inlier_mask]
                    pts_r_in = mkpts_r[inlier_mask]

                    # Розрахунок RMSE для оцінки якості геометрії
                    pts_q_transformed = GeometryTransforms.apply_homography(pts_q_in, H_eval)
                    rmse = float(
                        np.sqrt(np.mean(np.sum((pts_q_transformed - pts_r_in) ** 2, axis=1)))
                    )

                    if inliers > best_inliers and inliers >= self.min_matches:
                        best_inliers = inliers
                        best_candidate_id = candidate_id
                        best_H_query_to_ref = H_eval
                        best_mkpts_q_inliers = pts_q_in
                        best_mkpts_r_inliers = pts_r_in
                        best_total_matches = len(mkpts_q)
                        best_rmse = rmse
                        logger.debug(
                            f"Homography for {candidate_id}: {inliers} inliers, RMSE: {rmse:.2f}"
                        )

            if best_inliers >= early_stop:
                logger.info(
                    f"Early stop triggered with {best_inliers} inliers on candidate {best_candidate_id}"
                )
                break

        # Оскільки LightGlue (ALIKED) тепер основний метод, окремий fallback не потрібен

        if (
            best_inliers < self.min_matches
            or best_mkpts_r_inliers is None
            or best_H_query_to_ref is None
        ):
            # Спробуємо фоллбек перед тим як повертати помилку.
            # Якщо ми не знайшли жодного відповідного кадру через Matching, беремо топ-1 з retrieval.
            target_id = (
                best_candidate_id if (best_candidate_id != -1) else best_global_candidates[0][0]
            )
            fallback_res = self._localize_by_reference_frame(target_id, best_global_score)
            if fallback_res:
                logger.info(
                    f"Feature matching insufficient ({best_inliers} inliers < {self.min_matches} min), "
                    f"using retrieval-only fallback | "
                    f"frame={target_id}, global_score={best_global_score:.3f}"
                )
                return fallback_res
            logger.warning(
                f"Localization failed: {best_inliers} inliers < {self.min_matches} minimum | "
                f"best_candidate={best_candidate_id}, candidates_tried={len(best_global_candidates)}, "
                f"query_kpts={len(best_query_features.get('keypoints', []))}"
            )
            self._consecutive_failures += 1
            return {
                "success": False,
                "error": f"Not enough valid inliers ({best_inliers} < {self.min_matches})",
            }

        # 3. Отримуємо матрицю знайденого кадру з бази
        affine_ref = self.database.get_frame_affine(best_candidate_id)
        if affine_ref is None:
            target_id = (
                best_candidate_id if (best_candidate_id != -1) else best_global_candidates[0][0]
            )
            fallback_res = self._localize_by_reference_frame(target_id, best_global_score)
            if fallback_res:
                logger.info(
                    f"No propagated calibration for frame {target_id} — "
                    f"frame may not have been reached during calibration propagation. "
                    f"Using retrieval-only fallback."
                )
                return fallback_res
            return {
                "success": False,
                "error": (
                    f"No propagated calibration for matched frame {best_candidate_id}. "
                    f"Run calibration propagation to enable localization for this area."
                ),
            }

        # 4. Рахуємо розміри ПОВЕРНУТОГО зображення
        if best_global_angle in [90, 270]:
            rot_height, rot_width = width, height
        else:
            rot_height, rot_width = height, width

        # 4. Багатоточкова локалізація більше не потрібна. Беремо ідеальний центр кадру

        # Використовуємо знайдену Homography
        M_query_to_ref = best_H_query_to_ref
        if M_query_to_ref is None:
            return {"success": False, "error": "Failed to compute transform"}

        # 5. Трансформуємо центральну точку: Query -> Reference (через Homography) -> Metric (через Affine)
        center_query = np.array([[rot_width / 2.0, rot_height / 2.0]], dtype=np.float32)
        pts_in_ref = GeometryTransforms.apply_homography(center_query, M_query_to_ref)
        if pts_in_ref is None or len(pts_in_ref) == 0:
            target_id = (
                best_candidate_id if (best_candidate_id != -1) else best_global_candidates[0][0]
            )
            fallback_res = self._localize_by_reference_frame(target_id, best_global_score)
            if fallback_res:
                logger.info(
                    f"Homography transform failure, using retrieval-only fallback for frame {target_id} (score {best_global_score:.3f})"
                )
                return fallback_res
            return {
                "success": False,
                "error": "Coordinate transformation error (homography failed)",
            }

        pts_metric = GeometryTransforms.apply_affine(pts_in_ref, affine_ref)

        # Оскільки ми взяли одну центральну точку, просто беремо її координати
        mx = float(pts_metric[0, 0])
        my = float(pts_metric[0, 1])
        metric_pt = np.array([mx, my], dtype=np.float32)

        # 6. Перевіряємо чи нова точка — аномалія (стрибок координат)
        if self.outlier_detector.is_outlier(metric_pt, dt):
            logger.warning(
                f"Outlier filtered | matched_frame={best_candidate_id}, "
                f"metric=({mx:.1f}, {my:.1f}), inliers={best_inliers}, dt={dt:.3f}s. "
                f"Position jump was too large relative to recent trajectory."
            )
            return {"success": False, "error": "Outlier detected — position jump filtered"}

        # Успішна локалізація — скидаємо лічильник невдач
        self._consecutive_failures = 0

        # Оновлення Калмана (фільтрація шумів)
        filtered_pt = self.trajectory_filter.update(metric_pt, dt=dt)

        self.outlier_detector.add_position(filtered_pt, dt=dt)
        lat, lon = self.calibration.converter.metric_to_gps(
            float(filtered_pt[0]), float(filtered_pt[1])
        )

        # Зсув для корекції FOV через фільтрацію
        dx, dy = filtered_pt[0] - metric_pt[0], filtered_pt[1] - metric_pt[1]

        # 7. Розрахунок поля зору (FOV)
        # Користувач очікує бачити повне покриття камери (від 0 до rot_width).
        # Проектуємо повний кадр
        corners = np.array(
            [[0, 0], [rot_width, 0], [rot_width, rot_height], [0, rot_height]], dtype=np.float32
        )

        ref_corners = GeometryTransforms.apply_homography(corners, M_query_to_ref)

        # Захист від перспективного "вибуху" гомографії (якщо кластер ALIKED занадто локальний)
        is_exploded = False
        if ref_corners is not None:
            max_coord = np.max(np.abs(ref_corners))
            if max_coord > 50000:  # якщо кут відлетів далі ніж на 50к пікселів
                is_exploded = True

        if is_exploded and best_mkpts_q_inliers is not None and len(best_mkpts_q_inliers) > 0:
            logger.warning(
                f"Homography exploded the FOV (max_coord={max_coord:.0f}px > 50000px threshold). "
                f"Cause: perspective distortion from locally-clustered ALIKED matches. "
                f"Falling back to inliers bounding box for safe FOV estimation."
            )
            pts = best_mkpts_q_inliers
            min_x, min_y = np.min(pts, axis=0)
            max_x, max_y = np.max(pts, axis=0)
            pad_x, pad_y = (max_x - min_x) * 0.1, (max_y - min_y) * 0.1
            safe_corners = np.array(
                [
                    [max(0, min_x - pad_x), max(0, min_y - pad_y)],
                    [min(rot_width, max_x + pad_x), max(0, min_y - pad_y)],
                    [min(rot_width, max_x + pad_x), min(rot_height, max_y + pad_y)],
                    [max(0, min_x - pad_x), min(rot_height, max_y + pad_y)],
                ],
                dtype=np.float32,
            )
            ref_corners = GeometryTransforms.apply_homography(safe_corners, M_query_to_ref)
            original_poly_px = safe_corners
        else:
            original_poly_px = corners
            logger.debug("FOV projected using full frame Homography matrix.")

        # --- ДІАГНОСТИЧНІ ЛОГИ МАКСИМАЛЬНОГО РІВНЯ ---
        logger.info(f"--- FOV DIAGNOSTICS FOR FRAME {best_candidate_id} ---")
        w_px = np.linalg.norm(original_poly_px[0] - original_poly_px[1])
        h_px = np.linalg.norm(original_poly_px[1] - original_poly_px[2])
        logger.info(f"[1] Original FOV in Query image: {w_px:.1f} x {h_px:.1f} pixels")

        if ref_corners is not None:
            w_ref = np.linalg.norm(ref_corners[0] - ref_corners[1])
            h_ref = np.linalg.norm(ref_corners[1] - ref_corners[2])
            logger.info(
                f"[2] FOV mapped to Reference via Homography: {w_ref:.1f} x {h_ref:.1f} pixels"
            )

        gps_corners = []
        if ref_corners is not None:
            metric_corners = GeometryTransforms.apply_affine(ref_corners, affine_ref)
            if metric_corners is not None:
                # Діагностика розмірів FOV в метрах
                fov_w = np.linalg.norm(metric_corners[1] - metric_corners[0])
                fov_h = np.linalg.norm(metric_corners[3] - metric_corners[0])
                logger.info(
                    f"[3] FOV mapped to Web Mercator Metric space: {fov_w:.1f}m x {fov_h:.1f}m"
                )
                logger.debug(
                    f"FOV dimensions: {fov_w:.1f}m x {fov_h:.1f}m | "
                    f"Center metric: ({mx:.1f}, {my:.1f}) | "
                    f"Filtered: ({filtered_pt[0]:.1f}, {filtered_pt[1]:.1f})"
                )
                for cx, cy in metric_corners:
                    try:
                        clat, clon = self.calibration.converter.metric_to_gps(
                            float(cx + dx), float(cy + dy)
                        )
                        gps_corners.append((clat, clon))
                    except Exception:
                        pass

        confidence = self._compute_confidence(
            best_candidate_id, best_inliers, best_total_matches, best_rmse
        )

        # ДІАГНОСТИКА
        logger.debug(
            f"Localize Frame {best_candidate_id}: Center transformed via Homography (8 DoF)"
        )
        logger.debug(f"Sample Center METRIC: ({mx:.1f}, {my:.1f})")

        logger.success(
            f"Localized ({lat:.6f}, {lon:.6f}) | frame={best_candidate_id} | "
            f"metric=({mx:.1f}, {my:.1f}) | inliers={best_inliers} | conf={confidence:.2f}"
        )

        return {
            "success": True,
            "lat": lat,
            "lon": lon,
            "confidence": confidence,
            "matched_frame": int(best_candidate_id),
            "inliers": int(best_inliers),
            "fov_polygon": gps_corners,
            "sample_spread_m": 0.0,
        }

    def _compute_confidence(
        self, best_candidate_id: int, best_inliers: int, total_matches: int, rmse_val: float
    ) -> float:
        """Обчислює впевненість на основі QA бази даних (RMSE, Disagreement) та кількості інлаєрів."""
        # Налаштування з конфігу
        max_inliers = get_cfg(self.config, "localization.confidence.confidence_max_inliers", 80)
        rmse_norm = get_cfg(self.config, "localization.confidence.rmse_norm_m", 10.0)
        diag_norm = get_cfg(self.config, "localization.confidence.disagreement_norm_m", 5.0)
        w_inlier = get_cfg(self.config, "localization.confidence.inlier_weight", 0.7)
        w_stability = get_cfg(self.config, "localization.confidence.stability_weight", 0.3)

        # 1. Показник інлаєрів (0-1)
        inlier_score = min(1.0, best_inliers / max_inliers)

        # 2. Показник стабільності бази (на основі QA метрик)
        rmse = (
            self.database.frame_rmse[best_candidate_id]
            if self.database.frame_rmse is not None
            else 0.0
        )
        disagreement = (
            self.database.frame_disagreement[best_candidate_id]
            if self.database.frame_disagreement is not None
            else 0.0
        )

        stability_score = 1.0 - (
            min(rmse, rmse_norm) / rmse_norm * 0.5 + min(disagreement, diag_norm) / diag_norm * 0.5
        )
        stability_score = float(np.clip(stability_score, 0.0, 1.0))

        # 3. Показник поточної відповідності (ПЕР-ФРЕЙМ)
        # a) Inlier ratio
        ratio_score = float(best_inliers / (total_matches + 1e-6))
        # b) RMSE score (1.0 if RMSE=0, 0.5 if RMSE=thresh)
        rmse_score_val = 1.0 / (1.0 + (rmse_val / (self.ransac_thresh + 1e-6)))

        match_score = ratio_score * 0.5 + rmse_score_val * 0.5

        # 4. Комбінована оцінка
        # (QA бази * 0.3) + (Кількість інлаєрів * 0.4) + (Якість відповідності * 0.3)
        final_conf = stability_score * 0.3 + inlier_score * 0.4 + match_score * 0.3

        return float(np.clip(final_conf, 0.05, 1.0))

    def _localize_by_reference_frame(self, frame_id: int, score: float) -> dict:
        """Приблизна локалізація за центром опорного кадру (retrieval-only fallback)"""
        if frame_id == -1:
            return None

        threshold = get_cfg(self.config, "localization.retrieval_only_min_score", 0.90)
        if score < threshold:
            logger.debug(
                f"Retrieval-only fallback rejected: score {score:.3f} < threshold {threshold:.3f} | "
                f"frame={frame_id}"
            )
            return None

        affine_ref = self.database.get_frame_affine(frame_id)
        if affine_ref is None:
            logger.debug(
                f"Retrieval-only fallback failed: no affine matrix for frame {frame_id}. "
                f"Frame not reached during calibration propagation."
            )
            return None

        ref_h, ref_w = self.database.get_frame_size(frame_id)
        # Центр кадру в системі координат БД
        center_ref = np.array([[ref_w / 2, ref_h / 2]], dtype=np.float32)
        metric_pt = GeometryTransforms.apply_affine(center_ref, affine_ref)[0]

        lat, lon = self.calibration.converter.metric_to_gps(metric_pt[0], metric_pt[1])

        return {
            "success": True,
            "lat": lat,
            "lon": lon,
            "confidence": 0.3,  # Низький confidence для retrieval-only
            "inliers": 0,
            "matched_frame": frame_id,
            "fallback_mode": "retrieval_only",
            "global_score": score,
            "fov_polygon": None,
        }
