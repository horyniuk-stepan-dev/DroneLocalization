import os
import time

import numpy as np

from config.config import get_cfg
from src.geometry.transformations import GeometryTransforms
from src.localization.matcher import FastRetrieval, LanceDBRetrieval
from src.tracking.kalman_filter import TrajectoryFilter
from src.tracking.outlier_detector import OutlierDetector
from src.utils.logging_utils import get_logger
from src.utils.resolution_normalizer import ResolutionNormalizer
from src.utils.telemetry import Telemetry

logger = get_logger(__name__)

FAILURE_TYPES = {
    "out_of_coverage": "out_of_coverage",
    "No candidates": "no_retrieval_candidates",
    "Not enough valid inliers": "insufficient_inliers",
    "No propagated calibration": "no_propagated_affine",
    "Outlier detected": "trajectory_outlier",
    "Coordinate transformation": "transform_error",
}

# Матриці обертання вектора зсуву для кожного кута повороту кадру.
# np.rot90(frame, k=K) повертає кадр проти годинникової стрілки на K*90°.
# Якщо трекер обчислив зсув (dx, dy) в оригінальній системі координат,
# цей словник перераховує його в систему координат повернутого кадру,
# де побудована гомографія H.
#
# Виведення:
#   k=1 (90° CCW):  x_rot = -y_orig, y_rot = x_orig  →  (dx,dy) → (-dy, dx)
#   k=2 (180°):     x_rot = -x_orig, y_rot = -y_orig →  (dx,dy) → (-dx,-dy)
#   k=3 (270° CCW): x_rot = y_orig,  y_rot = -x_orig →  (dx,dy) → (dy, -dx)
_ROTATION_VEC: dict[int, tuple[int, int, int, int]] = {
    # angle: (a, b, c, d) → new_dx = a*dx + b*dy, new_dy = c*dx + d*dy
    0:   ( 1,  0,  0,  1),
    90:  ( 0, -1,  1,  0),
    180: (-1,  0,  0, -1),
    270: ( 0,  1, -1,  0),
}


class Localizer:
    def __init__(
        self,
        database,
        feature_extractor,
        matcher,
        calibration,
        config=None,
        ref_frame_width: int = 0,
        ref_frame_height: int = 0,
        db_manager=None,
        calib_manager=None,
    ):
        self.database = database
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        self.calibration = calibration
        self.config = config or {}

        # Мультиджерельна підтримка (Phase 3 ТЗ)
        self.db_manager = db_manager        # MultiDatabaseManager | None
        self.calib_manager = calib_manager  # MultiCalibrationManager | None
        self._active_source_id: str | None = None

        # Дефолти синхронізовані з APP_CONFIG через get_cfg()
        self.min_matches = get_cfg(self.config, "localization.min_matches", 12)
        self.ransac_thresh = get_cfg(self.config, "localization.ransac_threshold", 3.0)
        self.enable_auto_rotation = get_cfg(self.config, "localization.auto_rotation", True)
        self.homography_backend = get_cfg(self.config, "homography.backend", "opencv")
        self.use_mad_ransac = get_cfg(self.config, "homography.use_mad_ransac", True)
        self.mad_k_factor = get_cfg(self.config, "homography.mad_k_factor", 2.5)

        self.trajectory_filter = TrajectoryFilter(
            process_noise=get_cfg(self.config, "tracking.kalman_process_noise", 2.0),
            measurement_noise=get_cfg(self.config, "tracking.kalman_measurement_noise", 5.0),
            dt=1.0,
        )
        self.outlier_detector = OutlierDetector(
            window_size=get_cfg(self.config, "tracking.outlier_window", 10),
            threshold_std=get_cfg(self.config, "tracking.outlier_threshold_std", 150.0),
            max_speed_mps=get_cfg(self.config, "tracking.max_speed_mps", 1000.0),
            max_consecutive=get_cfg(self.config, "tracking.max_consecutive_outliers", 5),
        )

        # Retriever: при мульти-режимі він у db_manager, тут — для single-mode
        self.retriever = None
        if self.db_manager is None:
            # Single-database mode (зворотна сумісність)
            if hasattr(self.database, "lance_table") and self.database.lance_table is not None:
                self.retriever = LanceDBRetrieval(self.database.lance_table)
            else:
                self.retriever = FastRetrieval(self.database.global_descriptors)

        self.model_manager = self.config.get("_model_manager", None)
        self.fallback_enabled = get_cfg(self.config, "localization.enable_lightglue_fallback", True)
        self.min_inliers_for_accept = get_cfg(self.config, "localization.min_inliers_accept", 10)
        self.retrieval_top_k = get_cfg(self.config, "localization.retrieval_top_k", 8)
        self.early_stop_inliers = get_cfg(self.config, "localization.early_stop_inliers", 30)

        # Fix #1: Захист від нескінченного циклу при виході за межі покриття
        self._consecutive_failures = 0
        self._max_failures = get_cfg(self.config, "localization.max_consecutive_failures", 10)

        # Нормалізація роздільної здатності вхідного кадру до еталонної роздільної здатності БД
        self.normalizer = ResolutionNormalizer(ref_frame_width, ref_frame_height)
        self._last_scale = 1.0

        # ── Patchify: мультипатч-retrieval ────────────────────────────────────
        # ВАЖЛИВО: PatchifyRetrieval ініціалізується тільки ЯКЩО:
        #   1. Увімкнено через конфіг
        #   2. В базі є patch_descriptors (тобто БД будувалась з use_patchify=True)
        # Якщо хоча б одна умова не виконана — patchify мовчки вимкнено (backward compat).
        self.patchify_retrieval = None
        use_patchify = get_cfg(self.config, "localization.use_patchify", False)
        if use_patchify:
            patch_desc = getattr(self.database, "patch_descriptors", None)
            if patch_desc is not None and len(patch_desc) > 0:
                try:
                    from src.localization.patchify import PatchifyRetrieval
                    patchify_grids = get_cfg(
                        self.config, "localization.patchify_grids", [[1, 1], [2, 2], [3, 3]]
                    )
                    patchify_batch = get_cfg(
                        self.config, "localization.patchify_batch_size", 1
                    )
                    desc_dim = int(patch_desc.shape[-1])
                    self.patchify_retrieval = PatchifyRetrieval(
                        self.feature_extractor,
                        descriptor_dim=desc_dim,
                        grids=patchify_grids,
                        batch_size=patchify_batch,
                    )
                    frame_ids = list(range(self.database.get_num_frames()))
                    self.patchify_retrieval.build_index(patch_desc, frame_ids)
                    logger.info(
                        f"Patchify retrieval initialized: "
                        f"{self.patchify_retrieval.num_patches} patches/frame, "
                        f"dim={desc_dim}, grids={patchify_grids}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Patchify retrieval init failed — falling back to standard retrieval: {e}"
                    )
                    self.patchify_retrieval = None
            else:
                logger.info(
                    "Patchify enabled in config but database has no patch_descriptors. "
                )

        # Phase 3.2: GSD integration
        project_manager = self.config.get("_project_manager", None)
        if project_manager and project_manager.settings:
            try:
                from src.geometry.gsd_calculator import GSDCalculator
                s = project_manager.settings
                gsd = GSDCalculator(
                    altitude_m=getattr(s, "altitude_m", 100.0),
                    focal_length_mm=getattr(s, "focal_length_mm", 13.2),
                    sensor_width_mm=getattr(s, "sensor_width_mm", 8.8),
                    image_width_px=getattr(s, "image_width_px", 4000),
                )
                gsd.log_summary()
                self.calibration.set_gsd_calculator(gsd)
            except Exception as e:
                logger.warning(f"Failed to initialize GSD Calculator: {e}")

    # ─────────────────────────────────────────────────────────────────────────

    def localize_frame(
        self, query_frame: np.ndarray, static_mask: np.ndarray = None, dt: float = 1.0
    ) -> dict:
        # Fix #1: Якщо було занадто багато послідовних невдач — повертаємо out_of_coverage
        if self._consecutive_failures >= self._max_failures:
            self._consecutive_failures = 0
            self._log_failure(
                FAILURE_TYPES["out_of_coverage"],
                details=f"Exceeded {self._max_failures} failures",
            )
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

        # Нормалізація до еталонної роздільної здатності БД
        query_frame, self._last_scale = self.normalizer.normalize(query_frame)
        if static_mask is not None:
            static_mask = self.normalizer.normalize_mask(static_mask)
        height, width = query_frame.shape[:2]

        angles_to_try = [0, 90, 180, 270] if self.enable_auto_rotation else [0]

        best_global_score = -1.0
        best_global_angle = 0
        best_global_candidates = []

        top_k = self.retrieval_top_k

        # ── Крок 1: Вибір найкращого ракурсу за стандартним DINOv2 ──────────
        # ТІЛЬКИ стандартний retrieval — patchify тут не запускається, бо:
        #   - compute_patch_descriptors = 14 DINOv2 forward-пасів (для grids 1+4+9)
        #   - 4 кути × 14 патчів = 56 зайвих форвард-пасів лише на вибір кута
        #   - patchify-скор (avg cosine по 14 патчах) ≠ DINOv2 CLS-token cosine →
        #     змішування робить порівняння між кутами некоректним
        best_source_id_per_angle: str | None = None  # для мульти-режиму

        for angle in angles_to_try:
            k = angle // 90
            rotated_frame = np.rot90(query_frame, k=k).copy()
            global_desc = self.feature_extractor.extract_global_descriptor(rotated_frame)

            with Telemetry.profile("retrieval"):
                if self.db_manager is not None:
                    # Мульти-режим: пошук у всіх активних базах
                    src_id, candidates = self.db_manager.get_best_match(
                        global_desc, top_k=top_k
                    )
                else:
                    # Single-mode (зворотна сумісність)
                    candidates = self.retriever.find_similar_frames(
                        global_desc, top_k=top_k
                    )
                    src_id = None

            if candidates:
                top_score = candidates[0][1]
                if top_score > best_global_score:
                    best_global_score = top_score
                    best_global_angle = angle
                    best_global_candidates = candidates
                    best_source_id_per_angle = src_id

        if not best_global_candidates:
            self._consecutive_failures += 1
            self._log_failure(FAILURE_TYPES["No candidates"])
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

        # ── Крок 1.5a: Перемикання database/calibration для мульти-режиму ───
        if self.db_manager is not None and best_source_id_per_angle is not None:
            self._active_source_id = best_source_id_per_angle
            self.database = self.db_manager.get_database(best_source_id_per_angle)
            if self.calib_manager is not None:
                self.calibration = self.calib_manager.get(best_source_id_per_angle)
            logger.debug(f"Active source switched to '{best_source_id_per_angle}'")

        # ── Крок 1.5: Готуємо повернутий кадр для найкращого ракурсу ────────
        k = best_global_angle // 90
        best_rotated_frame = np.rot90(query_frame, k=k).copy()
        best_rotated_mask = (
            np.rot90(static_mask, k=k).copy() if static_mask is not None else None
        )

        # ── Крок 1.6: Patchify-розширення кандидатів (тільки для найкращого ракурсу) ─
        # Запускаємо ОДИН РАЗ після вибору кута — не в циклі.
        # Пatchify додає кандидатів, яких міг пропустити CLS-token DINOv2
        # (наприклад, при зміні висоти польоту).
        if self.patchify_retrieval is not None:
            try:
                with Telemetry.profile("patchify_retrieval"):
                    patch_descs = self.patchify_retrieval.compute_patch_descriptors(
                        best_rotated_frame
                    )
                    patch_candidates = self.patchify_retrieval.search(
                        patch_descs, top_k=top_k
                    )
                if patch_candidates:
                    # Об'єднуємо: max_results обмежує список, щоб не збільшувати час матчінгу
                    merged = self._merge_candidates(
                        best_global_candidates,
                        patch_candidates,
                        max_results=top_k * 2,
                    )
                    logger.debug(
                        f"Patchify expanded candidates: "
                        f"{len(best_global_candidates)} → {len(merged)} "
                        f"(top patchify score: {patch_candidates[0][1]:.3f})"
                    )
                    best_global_candidates = merged
            except Exception as e:
                logger.warning(f"Patchify retrieval failed, using standard candidates: {e}")

        # ── Крок 2: Локальна екстракція (ALIKED/RDD) для найкращого ракурсу ─
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

        # ── Крок 3: Локальний матчинг + RANSAC ──────────────────────────────
        for candidate_id, score in best_global_candidates:
            logger.debug(f"  → Trying candidate {candidate_id} (global_score={score:.3f})")
            ref_features = self.database.get_local_features(candidate_id)

            with Telemetry.profile("match"):
                mkpts_q, mkpts_r = self.matcher.match(best_query_features, ref_features)

            if len(mkpts_q) >= self.min_matches:
                with Telemetry.profile("ransac_homography"):
                    H_eval, mask = GeometryTransforms.estimate_homography(
                        mkpts_q,
                        mkpts_r,
                        ransac_threshold=self.ransac_thresh,
                        backend=self.homography_backend,
                        use_mad_ransac=self.use_mad_ransac,
                        mad_k_factor=self.mad_k_factor,
                    )

                if H_eval is not None:
                    inlier_mask = mask.ravel().astype(bool)
                    inliers = int(np.sum(inlier_mask))
                    pts_q_in = mkpts_q[inlier_mask]
                    pts_r_in = mkpts_r[inlier_mask]

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

        if (
            best_inliers < self.min_matches
            or best_mkpts_r_inliers is None
            or best_H_query_to_ref is None
        ):
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
            self._log_failure(FAILURE_TYPES["Not enough valid inliers"], inliers=best_inliers)
            return {
                "success": False,
                "error": f"Not enough valid inliers ({best_inliers} < {self.min_matches})",
            }

        # ── Крок 4: Отримуємо аффінну матрицю кандидата ─────────────────────
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
            self._log_failure(FAILURE_TYPES["No propagated calibration"])
            return {
                "success": False,
                "error": (
                    f"No propagated calibration for matched frame {best_candidate_id}. "
                    f"Run calibration propagation to enable localization for this area."
                ),
            }

        # Розміри повернутого нормалізованого зображення
        if best_global_angle in (90, 270):
            rot_height, rot_width = width, height
        else:
            rot_height, rot_width = height, width

        M_query_to_ref = best_H_query_to_ref
        if M_query_to_ref is None:
            self._log_failure(
                FAILURE_TYPES["Coordinate transformation"], details="Failed to compute transform"
            )
            return {"success": False, "error": "Failed to compute transform"}

        # ── Крок 5: Зберігаємо стан для Optical Flow ────────────────────────
        self._last_state = {
            "H": M_query_to_ref,
            "affine": affine_ref,
            "candidate_id": best_candidate_id,
            "inliers": best_inliers,
            "global_angle": best_global_angle,
            "source_id": self._active_source_id,
        }

        # ── Крок 6: Query center → Reference → Metric → GPS ─────────────────
        center_query = np.array([[rot_width / 2.0, rot_height / 2.0]], dtype=np.float64)
        pts_in_ref = GeometryTransforms.apply_homography(center_query, M_query_to_ref)
        if pts_in_ref is None or len(pts_in_ref) == 0:
            target_id = (
                best_candidate_id if (best_candidate_id != -1) else best_global_candidates[0][0]
            )
            fallback_res = self._localize_by_reference_frame(target_id, best_global_score)
            if fallback_res:
                logger.info(
                    f"Homography transform failure, using retrieval-only fallback for "
                    f"frame {target_id} (score {best_global_score:.3f})"
                )
                return fallback_res
            self._log_failure(FAILURE_TYPES["Coordinate transformation"])
            return {
                "success": False,
                "error": "Coordinate transformation error (homography failed)",
            }

        pts_metric = GeometryTransforms.apply_affine(pts_in_ref, affine_ref)
        mx = float(pts_metric[0, 0])
        my = float(pts_metric[0, 1])
        metric_pt = np.array([mx, my], dtype=np.float64)

        # ── Крок 7: Фільтрація аномалій ─────────────────────────────────────
        if self.outlier_detector.is_outlier(metric_pt, dt):
            logger.warning(
                f"Outlier filtered | matched_frame={best_candidate_id}, "
                f"metric=({mx:.1f}, {my:.1f}), inliers={best_inliers}, dt={dt:.3f}s. "
                f"Position jump was too large relative to recent trajectory."
            )
            self._log_failure(FAILURE_TYPES["Outlier detected"], inliers=best_inliers)
            return {"success": False, "error": "Outlier detected — position jump filtered"}

        self._consecutive_failures = 0

        filtered_pt = self.trajectory_filter.update(metric_pt, dt=dt)
        self.outlier_detector.add_position(filtered_pt, dt=dt)
        lat, lon = self.calibration.converter.metric_to_gps(
            float(filtered_pt[0]), float(filtered_pt[1])
        )
        dx, dy = filtered_pt[0] - metric_pt[0], filtered_pt[1] - metric_pt[1]

        # ── Крок 8: Розрахунок FOV ───────────────────────────────────────────
        corners = np.array(
            [[0, 0], [rot_width, 0], [rot_width, rot_height], [0, rot_height]],
            dtype=np.float32,
        )
        ref_corners = GeometryTransforms.apply_homography(corners, M_query_to_ref)

        is_exploded = False
        if ref_corners is not None:
            max_coord = np.max(np.abs(ref_corners))
            if max_coord > 50000:
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

        logger.debug(f"Localize Frame {best_candidate_id}: Center transformed via Homography (8 DoF)")
        logger.debug(f"Sample Center METRIC: ({mx:.1f}, {my:.1f})")
        source_str = f" | source={self._active_source_id}" if self._active_source_id else ""
        logger.success(
            f"Localized ({lat:.6f}, {lon:.6f}) | frame={best_candidate_id}{source_str} | "
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
            "source_id": self._active_source_id,
        }

    # ─────────────────────────────────────────────────────────────────────────

    def localize_optical_flow(
        self, dx_px: float, dy_px: float, dt: float, rot_width: int, rot_height: int
    ) -> dict:
        """Локалізація на основі піксельного зсуву від Optical Flow.

        Параметри rot_width / rot_height — ОРИГІНАЛЬНІ розміри кадру (до нормалізації
        і повороту), так як передаються з TrackingWorker через frame.shape.
        Метод самостійно перераховує їх у простір гомографії H:
          1. Масштабування на _last_scale (нормалізація роздільної здатності).
          2. Swap width↔height при 90° / 270° обертанні.
          3. Обертання вектора зсуву (dx, dy) у систему координат повернутого кадру.
        """
        if (
            not hasattr(self, "_last_state")
            or self._last_state["H"] is None
            or self._last_state["affine"] is None
        ):
            return {"success": False, "error": "No previous state to apply OF"}

        # Відновлюємо database/calibration для збереженого source_id (мульти-режим)
        last_source_id = self._last_state.get("source_id")
        if last_source_id is not None and self.db_manager is not None:
            self.database = self.db_manager.get_database(last_source_id)
            if self.calib_manager is not None:
                self.calibration = self.calib_manager.get(last_source_id)

        scale = self._last_scale
        angle = self._last_state.get("global_angle", 0)

        # ── 1. Вектор зсуву: оригінальний простір → нормалізований + повернутий ──
        # Масштабуємо до нормалізованого простору
        sdx = dx_px * scale
        sdy = dy_px * scale

        # Обертаємо вектор зсуву відповідно до повороту кадру.
        # H побудована в просторі повернутого нормалізованого кадру, тому зсув
        # теж має бути в тій самій системі координат.
        a, b, c, d = _ROTATION_VEC.get(angle, (1, 0, 0, 1))
        rot_sdx = a * sdx + b * sdy
        rot_sdy = c * sdx + d * sdy

        # ── 2. Розміри кадру: оригінальні → нормалізовані + повернуті ────────
        if angle in (90, 270):
            # 90° / 270°: рядки і стовпці міняються місцями
            norm_rot_w = rot_height * scale
            norm_rot_h = rot_width * scale
        else:
            norm_rot_w = rot_width * scale
            norm_rot_h = rot_height * scale

        # ── 3. Центр поточного кадру в системі координат попереднього ────────
        # Якщо з моменту KF точки змістились на (dx, dy), то центр поточного
        # дрона відповідає точці (center − displacement) у КС попереднього KF.
        center_query_shifted = np.array(
            [[norm_rot_w / 2.0 - rot_sdx, norm_rot_h / 2.0 - rot_sdy]],
            dtype=np.float64,
        )

        pts_in_ref = GeometryTransforms.apply_homography(
            center_query_shifted, self._last_state["H"]
        )
        if pts_in_ref is None or len(pts_in_ref) == 0:
            return {"success": False, "error": "OF homography failed"}

        pts_metric = GeometryTransforms.apply_affine(pts_in_ref, self._last_state["affine"])
        if pts_metric is None or len(pts_metric) == 0:
            return {"success": False, "error": "OF affine failed"}

        mx, my = float(pts_metric[0, 0]), float(pts_metric[0, 1])
        metric_pt = np.array([mx, my], dtype=np.float64)

        filtered_pt = self.trajectory_filter.update(metric_pt, dt=dt)
        self.outlier_detector.add_position(filtered_pt, dt=dt, reset_consecutive=False)

        lat, lon = self.calibration.converter.metric_to_gps(
            float(filtered_pt[0]), float(filtered_pt[1])
        )

        of_inliers = int(self._last_state.get("inliers", 30) * 0.8)

        return {
            "success": True,
            "lat": lat,
            "lon": lon,
            "confidence": 0.8,
            "matched_frame": int(self._last_state.get("candidate_id", -1)),
            "inliers": of_inliers,
            "fov_polygon": None,
            "is_of": True,
        }

    # ─────────────────────────────────────────────────────────────────────────

    def _merge_candidates(
        self,
        standard: list[tuple[int, float]],
        patches: list[tuple[int, float]],
        max_results: int | None = None,
    ) -> list[tuple[int, float]]:
        """Об'єднує результати стандартного та патч-retrieval через зважену суму.

        Ваги беруться з конфігу (localization.patchify_merge_weight).
        Якщо кадр є в обох джерелах: score = w_std * s + w_patch * p.
        Якщо тільки в одному: беремо скор як є (не штрафуємо за відсутність в іншому).
        """
        w_patch = get_cfg(self.config, "localization.patchify_merge_weight", 0.4)
        w_standard = 1.0 - w_patch

        standard_dict = dict(standard)
        patch_dict = dict(patches)
        all_fids = set(standard_dict.keys()) | set(patch_dict.keys())

        merged = {}
        for fid in all_fids:
            s = standard_dict.get(fid, 0.0)
            p = patch_dict.get(fid, 0.0)

            if fid in standard_dict and fid in patch_dict:
                merged[fid] = w_standard * s + w_patch * p
            elif fid in standard_dict:
                merged[fid] = s
            else:
                merged[fid] = p

        sorted_results = sorted(merged.items(), key=lambda x: x[1], reverse=True)

        if max_results is not None:
            sorted_results = sorted_results[:max_results]

        return sorted_results

    def _compute_confidence(
        self, best_candidate_id: int, best_inliers: int, total_matches: int, rmse_val: float
    ) -> float:
        """Обчислює впевненість на основі QA бази даних та кількості інлаєрів."""
        max_inliers = get_cfg(self.config, "localization.confidence.confidence_max_inliers", 80)
        rmse_norm = get_cfg(self.config, "localization.confidence.rmse_norm_m", 10.0)
        diag_norm = get_cfg(self.config, "localization.confidence.disagreement_norm_m", 5.0)
        w_inlier = get_cfg(self.config, "localization.confidence.inlier_weight", 0.7)
        w_stability = get_cfg(self.config, "localization.confidence.stability_weight", 0.3)

        inlier_score = min(1.0, best_inliers / max_inliers)

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
            min(rmse, rmse_norm) / rmse_norm * 0.5
            + min(disagreement, diag_norm) / diag_norm * 0.5
        )
        stability_score = float(np.clip(stability_score, 0.0, 1.0))

        ratio_score = float(best_inliers / (total_matches + 1e-6))
        rmse_score_val = 1.0 / (1.0 + (rmse_val / (self.ransac_thresh + 1e-6)))
        match_score = ratio_score * 0.5 + rmse_score_val * 0.5

        final_conf = stability_score * 0.3 + inlier_score * 0.4 + match_score * 0.3
        return float(np.clip(final_conf, 0.05, 1.0))

    def _localize_by_reference_frame(self, frame_id: int, score: float) -> dict:
        """Приблизна локалізація за центром опорного кадру (retrieval-only fallback)."""
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
        center_ref = np.array([[ref_w / 2, ref_h / 2]], dtype=np.float64)
        metric_pt = GeometryTransforms.apply_affine(center_ref, affine_ref)[0]

        lat, lon = self.calibration.converter.metric_to_gps(metric_pt[0], metric_pt[1])

        return {
            "success": True,
            "lat": lat,
            "lon": lon,
            "confidence": 0.3,
            "inliers": 0,
            "matched_frame": frame_id,
            "fallback_mode": "retrieval_only",
            "global_score": score,
            "fov_polygon": None,
        }

    def _log_failure(self, error_type: str, inliers: int = 0, details: str = ""):
        try:
            csv_path = "logs/localization_failures.csv"
            write_header = not os.path.exists(csv_path)
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            with open(csv_path, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("timestamp,error_type,inliers,details\n")
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                safe_details = details.replace('"', '""')
                f.write(f'{timestamp},{error_type},{inliers},"{safe_details}"\n')
        except Exception as e:
            logger.error(f"Failed to log to localization_failures.csv: {e}")
