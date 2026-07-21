import numpy as np

from config import get_cfg
from src.geometry.point_spread import inlier_spread
from src.geometry.transformations import GeometryTransforms
from src.localization.candidate_retriever import CandidateRetriever
from src.localization.failure_log import FAILURE_TYPES, FailureLogger
from src.localization.geometric_verifier import GeometricVerifier
from src.localization.matcher import FastRetrieval, LanceDBRetrieval
from src.localization.result_builder import ResultBuilder
from src.localization.rotation_geometry import _ROTATION_VEC, _rotate_point_np90
from src.localization.rotation_selector import RotationSelector
from src.localization.scale_manager import ScaleManager, crop_to_affine
from src.tracking.kalman_filter import TrajectoryFilter
from src.tracking.outlier_detector import OutlierDetector
from src.utils.logging_utils import get_logger
from src.utils.resolution_normalizer import ResolutionNormalizer
from src.utils.telemetry import Telemetry

logger = get_logger(__name__)


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
        self._failure_logger = FailureLogger()

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
            threshold_std=get_cfg(self.config, "tracking.outlier_threshold_std", 4.0),
            max_speed_mps=get_cfg(self.config, "tracking.max_speed_mps", 120.0),
            max_consecutive=get_cfg(self.config, "tracking.max_consecutive_outliers", 5),
        )

        # RESEARCH 3.1: ковзний віконний back-end smoother (флаг, дефолт off).
        # Синхронний 2D Huber-IRLS поверх keyframe-фіксів + OF-одометрії;
        # корекція KF зсувом. Див. src/tracking/smoother.py.
        self._smoother = None
        if get_cfg(self.config, "tracking.smoother_enabled", False):
            from src.tracking.smoother import SlidingWindowSmoother

            self._smoother = SlidingWindowSmoother(
                window=get_cfg(self.config, "tracking.smoother_window", 60),
                huber_k=get_cfg(self.config, "tracking.smoother_huber_k", 1.2),
                fix_sigma_base_m=get_cfg(
                    self.config, "tracking.smoother_fix_sigma_base_m", 5.0
                ),
                odom_sigma_base_m=get_cfg(
                    self.config, "tracking.smoother_odom_sigma_base_m", 3.0
                ),
                max_correction_m=get_cfg(
                    self.config, "tracking.smoother_max_correction_m", 50.0
                ),
                entry_prior_sigma_m=get_cfg(
                    self.config, "tracking.smoother_entry_prior_sigma_m", 15.0
                ),
                irls_iterations=get_cfg(
                    self.config, "tracking.smoother_irls_iterations", 4
                ),
                correction_lag=get_cfg(
                    self.config, "tracking.smoother_correction_lag", 10
                ),
                deadband_m=get_cfg(self.config, "tracking.smoother_deadband_m", 2.0),
                gain=get_cfg(self.config, "tracking.smoother_gain", 0.25),
                max_step_m=get_cfg(self.config, "tracking.smoother_max_step_m", 3.0),
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
        # RESEARCH 2.2: аварійний SIFT+LightGlue фолбек
        self._sift_fallback = get_cfg(self.config, "localization.sift_fallback", False)
        self._sift_fallback_max_cand = get_cfg(
            self.config, "localization.sift_fallback_max_candidates", 3
        )
        self.early_stop_inliers = get_cfg(self.config, "localization.early_stop_inliers", 30)

        # ADDENDUM 1.1: статистика розкиду інлаєрів. Без неї прогін не дає
        # вердикту — критерій приймання сформульований саме як ЧАСТОТА
        # спрацювання («< 1% keyframe-ів зі spread < 0.10 → пункт відкотити»).
        # Лічильники живуть лише коли увімкнено spread_confidence_enabled.
        self._spread_stats_enabled = get_cfg(
            self.config, "localization.spread_confidence_enabled", False
        )
        self._spread_log_every = get_cfg(self.config, "localization.spread_log_every", 50)
        self._spread_n = 0
        self._spread_n_low = 0
        self._spread_min = 1.0
        self._spread_sum = 0.0

        # Fix #1: Захист від нескінченного циклу при виході за межі покриття
        self._consecutive_failures = 0
        self._max_failures = get_cfg(self.config, "localization.max_consecutive_failures", 10)

        # Нормалізація роздільної здатності вхідного кадру до еталонної роздільної здатності БД
        self.normalizer = ResolutionNormalizer(ref_frame_width, ref_frame_height)
        self._last_scale = 1.0

        # A3: темпоральний prior на кут повороту — кут останньої успішної
        # локалізації; повний скан 4 кутів лише при просіданні score або невдачі
        self._last_best_angle: int | None = None

        # ── ScaleManager: GSD-ratio estimation for altitude-invariant localization ─
        self._scale_manager = ScaleManager(self.config)

        # Depth-based scale hint (soft pyramid reorder; hint only, never a hard scale).
        self._db_depth_scale = getattr(self.database, "median_depth_scale", None)
        self._use_depth_hint = get_cfg(self.config, "localization.scale_use_depth_hint", True)
        self._depth_hint_every_n = get_cfg(self.config, "localization.depth_hint_every_n", 30)
        self._depth_estimator = None
        self._depth_hint_counter = 0

        # ── Debug views: незалежний depth-інференс для вікна (окрема каденція) ─
        self._debug_depth_every_n = get_cfg(
            self.config, "debug_views.depth_every_n_keyframes", 1
        )
        self._debug_depth_estimator = None
        self._debug_depth_counter = 0

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

        self._candidate_retriever = CandidateRetriever(
            self.db_manager, self.retriever, self.patchify_retrieval, self.config
        )
        self._geometric_verifier = GeometricVerifier(
            self.matcher, self.min_matches, self.ransac_thresh,
            self.homography_backend, self.use_mad_ransac, self.mad_k_factor,
            self.early_stop_inliers,
            prefilter_enabled=get_cfg(self.config, "localization.candidate_prefilter", False),
            prefilter_keep=get_cfg(self.config, "localization.prefilter_keep", 2),
        )
        self._result_builder = ResultBuilder(self.config, self.ransac_thresh)
        self._rotation_selector = RotationSelector(
            self.feature_extractor, self._candidate_retriever, self.config
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

    @property
    def last_state(self) -> dict | None:
        """Останній успішний стан локалізації (H, affine, кут, source_id) або None.

        Публічний доступ замість читання приватного _last_state ззовні.
        """
        return getattr(self, "_last_state", None)

    def reset_session(self) -> None:
        """Скидає стан сесії трекінгу (фільтри, лічильники, кутовий prior).

        Викликати при старті нового відстеження, щоб уникнути хибних
        передбачень на основі попередньої сесії.
        """
        self.trajectory_filter.reset()
        self.outlier_detector.reset()
        self._consecutive_failures = 0
        self._last_best_angle = None
        self._last_state = None
        self._scale_manager.reset()
        self._debug_depth_counter = 0
        if self._smoother is not None:
            self._smoother.reset()

    def _maybe_set_depth_hint(self, frame: np.ndarray) -> None:
        """Soft depth-based reorder of the scale pyramid (every N keyframes; hint only)."""
        if not self._use_depth_hint or self._db_depth_scale is None:
            return
        self._depth_hint_counter += 1
        if (self._depth_hint_counter - 1) % max(1, self._depth_hint_every_n) != 0:
            return
        try:
            if self._depth_estimator is None:
                from src.depth.depth_estimator import DepthEstimator

                device = getattr(self.model_manager, "device", "cuda")
                self._depth_estimator = DepthEstimator.build(device=device)
            q_scale = self._depth_estimator.get_relative_scale(frame)
            self._scale_manager.set_depth_hint(q_scale, self._db_depth_scale)
        except Exception as e:
            logger.debug(f"Depth hint skipped: {e}")

    def _maybe_collect_depth(self, frame_rgb: np.ndarray, collector) -> None:
        """Debug: незалежний depth-інференс для вікна (окрема каденція).

        Не впливає на локалізацію — суто візуалізація «очима Depth Anything».
        Рахується лише коли вікно depth відкрите (collector.want_depth) і не
        частіше ніж кожен debug_views.depth_every_n_keyframes keyframe.
        """
        if collector is None or not collector.want_depth:
            return
        self._debug_depth_counter += 1
        if (self._debug_depth_counter - 1) % max(1, self._debug_depth_every_n) != 0:
            return
        try:
            if self._debug_depth_estimator is None:
                from src.depth.depth_estimator import DepthEstimator

                device = getattr(self.model_manager, "device", "cuda")
                self._debug_depth_estimator = DepthEstimator.build(device=device)
            depth = self._debug_depth_estimator.estimate(frame_rgb)
            collector.depth_map = depth
            # відносний масштаб з центру (як get_relative_scale, без 2-го інференсу)
            h, w = depth.shape
            cd = depth[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
            vm = cd > 0
            if bool(vm.any()):
                med = float(np.median(cd[vm]))
                collector.depth_scale = (1.0 / med) if med > 1e-6 else 1.0
            else:
                collector.depth_scale = 1.0
        except Exception as e:
            logger.debug(f"Debug depth skipped: {e}")

    def localize_frame(
        self,
        query_frame: np.ndarray,
        static_mask: np.ndarray = None,
        dt: float = 1.0,
        yaw_hint_deg: float | None = None,
        collector=None,
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
            self._last_best_angle = None  # наступний keyframe — повний скан кутів
            self._scale_manager.invalidate()  # повний скан масштабів теж
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

        # Depth hint: soft reorder of the scale pyramid toward the DB GSD (every N keyframes).
        self._maybe_set_depth_hint(query_frame)
        # Debug: depth-мапа для вікна (незалежно від успіху локалізації).
        self._maybe_collect_depth(query_frame, collector)

        angles_to_try = [0, 90, 180, 270] if self.enable_auto_rotation else [0]

        top_k = self.retrieval_top_k

        # ── RESEARCH 2.3: зовнішній yaw-hint (симулятор / телеметрія) ────────
        # yaw_hint_deg — кут CW у градусах, на який слід повернути кадр, щоб
        # він збігся з орієнтацією БД (north-up); конвертацію з курсу дрона
        # робить викликач. Квантуємо до 90° — весь rotation-тракт працює з
        # k·90. Хибний hint самовиліковується: якщо retrieval-score prior-кута
        # нижчий за rotation_rescan_min_score, RotationSelector сам виконає
        # повний батчований скан 4 кутів.
        prior_angle = self._last_best_angle
        use_prior = self.enable_auto_rotation and self._consecutive_failures == 0
        if yaw_hint_deg is not None and self.enable_auto_rotation:
            prior_angle = (int(round((yaw_hint_deg % 360.0) / 90.0)) * 90) % 360
            use_prior = True
            logger.debug(f"Yaw hint {yaw_hint_deg:.1f}° → prior rotation {prior_angle}°")

        rot = self._rotation_selector.select(
            query_frame,
            prior_angle,
            use_prior,
            angles_to_try,
            top_k,
            scale_manager=self._scale_manager,
        )
        if rot is None:
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
        best_global_score = rot.score
        best_global_angle = rot.angle
        best_global_candidates = rot.candidates
        best_source_id_per_angle = rot.source_id
        best_scale = rot.best_scale

        if collector is not None:
            collector.global_score = float(best_global_score)
            collector.global_angle = int(best_global_angle)
            collector.scale = float(best_scale)
            collector.retrieval_candidates = [
                (int(cid), float(sc)) for cid, sc in best_global_candidates
            ][: self.retrieval_top_k]

        logger.debug(
            f"Selected rotation {best_global_angle}° scale {best_scale:.2f} "
            f"with global score {best_global_score:.3f}"
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

        # ── Крок 1.5b: GSD-нормалізація (ScaleManager) ────────────────────────
        # Normalize the rotated frame to match DB's GSD before feature extraction.
        # In steady-state best_scale ≈ 1.0 and this is a no-op.
        _crop_info = None
        if abs(best_scale - 1.0) > 0.15:
            best_rotated_frame, _crop_info = self._scale_manager.normalize(
                best_rotated_frame, best_scale
            )
            if best_rotated_mask is not None:
                best_rotated_mask, _ = self._scale_manager.normalize(
                    best_rotated_mask, best_scale
                )
            logger.debug(
                f"GSD-normalized frame for scale {best_scale:.2f}: "
                f"{best_rotated_frame.shape[1]}x{best_rotated_frame.shape[0]}"
            )

        # ── Крок 1.6: Patchify-розширення кандидатів (тільки для найкращого ракурсу) ─
        # Запускаємо ОДИН РАЗ після вибору кута — не в циклі.
        # Пatchify додає кандидатів, яких міг пропустити CLS-token DINOv2
        # (наприклад, при зміні висоти польоту).
        best_global_candidates = self._candidate_retriever.expand(
            best_rotated_frame, best_global_candidates, top_k
        )

        # ── Крок 2: Локальна екстракція (ALIKED/RDD) для найкращого ракурсу ─
        best_query_features = self.feature_extractor.extract_local_features(
            best_rotated_frame, static_mask=best_rotated_mask
        )

        if collector is not None:
            collector.rotated_frame = best_rotated_frame
            collector.query_features = best_query_features
            if collector.want_dino_pca:
                try:
                    tokens, h_p, w_p = self.feature_extractor.extract_patch_tokens(
                        best_rotated_frame
                    )
                    collector.patch_tokens = tokens
                    collector.patch_grid = (h_p, w_p)
                except Exception as e:
                    logger.debug(f"Debug DINO tokens skipped: {e}")

        ver = self._geometric_verifier.verify(
            best_query_features, best_global_candidates, self.database
        )
        if ver is not None:
            best_inliers = ver.inliers
            best_candidate_id = ver.candidate_id
            best_H_query_to_ref = ver.H_query_to_ref
            best_mkpts_q_inliers = ver.mkpts_q_in
            best_mkpts_r_inliers = ver.mkpts_r_in
            best_total_matches = ver.total_matches
            best_rmse = ver.rmse
        else:
            best_inliers = 0
            best_candidate_id = -1
            best_H_query_to_ref = None
            best_mkpts_q_inliers = None
            best_mkpts_r_inliers = None
            best_total_matches = 0
            best_rmse = 999.0

        if collector is not None:
            collector.candidate_id = int(best_candidate_id)
            collector.inliers = int(best_inliers)
            collector.total_matches = int(best_total_matches)
            collector.rmse = float(best_rmse)
            collector.mkpts_q_inliers = best_mkpts_q_inliers
            collector.mkpts_r_inliers = best_mkpts_r_inliers

        # ADDENDUM 1.1: розкид інлаєрів — рахуємо ДО SIFT-фолбеку і
        # перераховуємо після нього, бо він підміняє набір точок.
        best_spread = self._inlier_spread(best_mkpts_q_inliers, best_query_features)
        if collector is not None:
            collector.spread = best_spread

        # ── RESEARCH 2.2: аварійний SIFT+LightGlue фолбек ────────────────────
        # ALIKED (як і SuperPoint) втрачає матчі при великому in-plane rotation
        # та екстремальній похилості [ISPRS 2025; MDPI RS 17(22)]. Одноразовий
        # перезапуск через ротаційно-інваріантний SIFT + LightGlue(sift) рятує
        # кадр до того, як він піде у retrieval-only фолбек.
        if (
            (best_inliers < self.min_matches or best_H_query_to_ref is None)
            and self._sift_fallback
            and getattr(self.database, "has_sift_features", False)
        ):
            rescue = self._try_sift_rescue(
                best_rotated_frame, best_rotated_mask, best_global_candidates
            )
            if rescue is not None:
                (
                    best_candidate_id,
                    best_H_query_to_ref,
                    best_inliers,
                    best_mkpts_q_inliers,
                    best_mkpts_r_inliers,
                    best_total_matches,
                    best_rmse,
                ) = rescue
                # Точки підмінені SIFT-ом — розкид більше не той, що вище.
                best_spread = self._inlier_spread(best_mkpts_q_inliers, best_query_features)
                if collector is not None:
                    collector.spread = best_spread

        self._record_spread(best_spread)

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

        # ── FOV-remap (IMPLEMENTATION_PLAN, Фаза 1.2) ────────────────────────────────────
        # H знайдена в координатах GSD-нормалізованого кадру (crop/resize).
        # Композиція з A (rotated→normalized) переводить H у координати
        # повернутого кадру — далі центр (Крок 6), FOV (Крок 8), OF-стан
        # (Крок 5) і scale-prior (update_from_homography) рахуються в одній
        # системі координат. Без цього при r < 0.85 центр зміщений на
        # ~(1−r)/2 кадру, полігон завищений у 1/r, а prior колапсує до 1.
        if _crop_info is not None and _crop_info.resize_scale != 1.0:
            n_h, n_w = best_rotated_frame.shape[:2]
            _A_norm = crop_to_affine(_crop_info, n_w, n_h)
            M_query_to_ref = M_query_to_ref @ _A_norm
            if best_mkpts_q_inliers is not None and len(best_mkpts_q_inliers) > 0:
                # mkpts лишаються в нормалізованих координатах лише для
                # collector (він малює по нормалізованому кадру); для
                # build_fov (кламп до rot_width/rot_height) переводимо в
                # координати повернутого кадру.
                _A_inv = crop_to_affine(_crop_info, n_w, n_h, inverse=True)
                best_mkpts_q_inliers = GeometryTransforms.apply_homography(
                    np.asarray(best_mkpts_q_inliers, dtype=np.float64), _A_inv
                )

        # ── Крок 5: Стан для Optical Flow (коміт — ПІСЛЯ outlier-гейту) ─────
        pending_state = {
            "H": M_query_to_ref,
            "affine": affine_ref,
            "candidate_id": best_candidate_id,
            "inliers": best_inliers,
            "global_angle": best_global_angle,
            "source_id": self._active_source_id,
            # Масштаб нормалізації САМЕ ЦЬОГО keyframe: OF має працювати в
            # системі кадру, якому належить H (свіжий self._last_scale на
            # наступних кадрах може вже відрізнятись).
            "scale": self._last_scale,
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
            # RESEARCH 3.1: відхилений фікс усе одно входить у вікно
            # smoother-а — Huber-вага арбітрує замість бінарного відкидання
            # (страхує Z-score false positives на різких маневрах).
            if self._smoother is not None:
                conf_rej = self._compute_confidence(
                    best_candidate_id, best_inliers, best_total_matches, best_rmse, best_spread
                )
                self._smoother.add_fix(
                    metric_pt,
                    dt=dt,
                    confidence=conf_rej,
                    source_id=self._active_source_id,
                    accepted=False,
                )
            return {"success": False, "error": "Outlier detected — position jump filtered"}

        # БАГФІКС (OF-шов): коміт стану лише ПІСЛЯ outlier-гейту. Раніше стан
        # комітився на Кроці 5 — і для відхилених кадрів, і до
        # homography-failure return — тож OF отримував H, неузгоджену з
        # prev_pts воркера (він не ребейзить точки без success).
        self._last_state = pending_state
        self._consecutive_failures = 0

        # Confidence рахуємо ДО фільтрації — B2: адаптивний шум вимірювання,
        # слабка локалізація впливає на траєкторію менше, впевнена — більше
        confidence = self._compute_confidence(
            best_candidate_id, best_inliers, best_total_matches, best_rmse, best_spread
        )

        filtered_pt = self.trajectory_filter.update(
            metric_pt, dt=dt, noise_scale=1.0 / max(confidence, 0.25)
        )
        # RESEARCH 3.1: back-end smoother — вікно фіксів + OF-одометрії;
        # корекція KF зсувом ДО запису в історію детектора та GPS/FOV,
        # щоб виправлення потрапило в ЦЕЙ же кадр.
        if self._smoother is not None:
            corr = self._smoother.add_fix(
                metric_pt,
                dt=dt,
                confidence=confidence,
                source_id=self._active_source_id,
                accepted=True,
                kf_xy=filtered_pt,
            )
            if corr is not None:
                self.trajectory_filter.shift(float(corr[0]), float(corr[1]))
                filtered_pt = (
                    float(filtered_pt[0]) + float(corr[0]),
                    float(filtered_pt[1]) + float(corr[1]),
                )
                logger.debug(
                    f"Smoother correction applied: ({corr[0]:+.2f}, {corr[1]:+.2f}) m"
                )
        self.outlier_detector.add_position(filtered_pt, dt=dt)
        lat, lon = self.calibration.converter.metric_to_gps(
            float(filtered_pt[0]), float(filtered_pt[1])
        )
        dx, dy = filtered_pt[0] - metric_pt[0], filtered_pt[1] - metric_pt[1]

        # ── Крок 8: Розрахунок FOV ───────────────────────────────────────────
        gps_corners = self._result_builder.build_fov(
            M_query_to_ref, affine_ref, rot_width, rot_height, best_mkpts_q_inliers,
            self.calibration.converter, dx, dy, mx, my, filtered_pt, best_candidate_id,
        )

        logger.debug(f"Localize Frame {best_candidate_id}: Center transformed via Homography (8 DoF)")
        logger.debug(f"Sample Center METRIC: ({mx:.1f}, {my:.1f})")
        source_str = f" | source={self._active_source_id}" if self._active_source_id else ""
        logger.success(
            f"Localized ({lat:.6f}, {lon:.6f}) | frame={best_candidate_id}{source_str} | "
            f"metric=({mx:.1f}, {my:.1f}) | inliers={best_inliers} | conf={confidence:.2f}"
        )

        # A3: запам'ятовуємо кут для темпорального prior наступного keyframe
        self._last_best_angle = best_global_angle

        # Scale prior: extract scale from H for the next keyframe
        self._scale_manager.update_from_homography(
            M_query_to_ref, rot_width, rot_height
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
        self,
        dx_px: float,
        dy_px: float,
        dt: float,
        rot_width: int,
        rot_height: int,
        flow_affine: np.ndarray | None = None,
        flow_quality: float | None = None,
    ) -> dict:
        """Локалізація на основі піксельного зсуву від Optical Flow.

        Параметри rot_width / rot_height — ОРИГІНАЛЬНІ розміри кадру (до нормалізації
        і повороту), так як передаються з TrackingWorker через frame.shape.
        Метод самостійно перераховує їх у простір гомографії H:
          1. Масштабування на _last_scale (нормалізація роздільної здатності).
          2. Swap width↔height при 90° / 270° обертанні.
          3. Обертання вектора зсуву (dx, dy) у систему координат повернутого кадру.

        B4: flow_affine — опційна симілярність 2x3 (original px, KF→current),
        оцінена по flow-точках. Враховує обертання/зміну масштабу між
        keyframe-ами (чиста трансляція dx/dy дрейфує на віражах).
        flow_quality (0..1) — чесна якість OF для адаптивного шуму Kalman.
        """
        state = self.last_state
        if state is None or state.get("H") is None or state.get("affine") is None:
            return {"success": False, "error": "No previous state to apply OF"}

        # Відновлюємо database/calibration для збереженого source_id (мульти-режим)
        last_source_id = self._last_state.get("source_id")
        if last_source_id is not None and self.db_manager is not None:
            self.database = self.db_manager.get_database(last_source_id)
            if self.calib_manager is not None:
                self.calibration = self.calib_manager.get(last_source_id)

        # Масштаб зі збереженого стану keyframe-а (узгоджений з його H);
        # фолбек на _last_scale для станів, записаних до цього поля.
        scale = self._last_state.get("scale", self._last_scale)
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
        center_query_shifted = None

        if flow_affine is not None:
            # B4: повна симілярність S (original px, KF→current). Точка KF-кадру,
            # що зараз опинилась у центрі: p0 = S⁻¹ @ center. Далі p0 переводимо
            # normalized → rotated (та сама трансформація, що й для кадру).
            try:
                S3 = np.vstack([np.asarray(flow_affine, dtype=np.float64), [0.0, 0.0, 1.0]])
                S_inv = np.linalg.inv(S3)
                cx0, cy0 = rot_width / 2.0, rot_height / 2.0
                p0x = S_inv[0, 0] * cx0 + S_inv[0, 1] * cy0 + S_inv[0, 2]
                p0y = S_inv[1, 0] * cx0 + S_inv[1, 1] * cy0 + S_inv[1, 2]
                # original → normalized
                p0x *= scale
                p0y *= scale
                # normalized → rotated (мапінг точки np.rot90, верифікований)
                w_n, h_n = rot_width * scale, rot_height * scale
                rx, ry = _rotate_point_np90(p0x, p0y, w_n, h_n, angle)
                center_query_shifted = np.array([[rx, ry]], dtype=np.float64)
            except np.linalg.LinAlgError:
                center_query_shifted = None  # вироджена S → fallback на трансляцію

        if center_query_shifted is None:
            # Fallback: чиста трансляція — якщо з моменту KF точки змістились на
            # (dx, dy), центр відповідає точці (center − displacement) у КС KF.
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

        # RESEARCH 3.1: сирий OF-фікс у вікно smoother-а — відносна одометрія,
        # прив'язана до H останнього прийнятого keyframe.
        if self._smoother is not None:
            self._smoother.note_of(metric_pt, dt=dt, quality=flow_quality)

        # B2: чесний confidence OF (раніше хардкод 0.8) + більший шум вимірювання
        # для Kalman (OF — відносне вимірювання, воно дрейфує від KF)
        if flow_quality is not None:
            of_conf = 0.5 + 0.35 * float(np.clip(flow_quality, 0.0, 1.0))
        else:
            of_conf = 0.7

        filtered_pt = self.trajectory_filter.update(
            metric_pt, dt=dt, noise_scale=1.5 / max(of_conf, 0.25)
        )
        self.outlier_detector.add_position(filtered_pt, dt=dt, reset_consecutive=False)

        lat, lon = self.calibration.converter.metric_to_gps(
            float(filtered_pt[0]), float(filtered_pt[1])
        )

        of_inliers = int(self._last_state.get("inliers", 30) * 0.8)

        return {
            "success": True,
            "lat": lat,
            "lon": lon,
            "confidence": round(of_conf, 3),
            "matched_frame": int(self._last_state.get("candidate_id", -1)),
            "inliers": of_inliers,
            "fov_polygon": None,
            "is_of": True,
        }

    # ─────────────────────────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        best_candidate_id: int,
        best_inliers: int,
        total_matches: int,
        rmse_val: float,
        spread: float | None = None,
    ) -> float:
        return self._result_builder.compute_confidence(
            best_candidate_id,
            best_inliers,
            total_matches,
            rmse_val,
            self.database,
            spread=spread,
        )

    def _record_spread(self, spread: float | None) -> None:
        """Накопичує статистику розкиду і періодично друкує її в лог.

        LOW_SPREAD = 0.10 — поріг із критерію приймання (≈ третина рівномірного
        покриття 0.289), а НЕ поріг штрафу (той — ``spread_ref`` = 0.15).
        """
        if not self._spread_stats_enabled or spread is None:
            return
        self._spread_n += 1
        self._spread_sum += spread
        self._spread_min = min(self._spread_min, spread)
        if spread < 0.10:
            self._spread_n_low += 1
        if self._spread_log_every > 0 and self._spread_n % self._spread_log_every == 0:
            pct = 100.0 * self._spread_n_low / self._spread_n
            logger.info(
                f"[spread] keyframes={self._spread_n} | spread<0.10: "
                f"{self._spread_n_low} ({pct:.1f}%) | mean={self._spread_sum / self._spread_n:.3f} "
                f"| min={self._spread_min:.3f} (норма ≈0.29; <1% → пункт 1.1 відкотити)"
            )

    @staticmethod
    def _inlier_spread(pts_q: np.ndarray | None, query_features: dict) -> float | None:
        """ADDENDUM 1.1: розкид інлаєрів у системі координат query-кадру.

        Розміри беремо з ``query_features["image_size"]`` (= [H, W] кадру, з
        якого екстрагувались фічі), а не з ``frame.shape``: keypoints живуть
        саме в цьому просторі — після ротації та scale-нормалізації, але вже
        відмасштабовані назад із ``max_local_edge`` (feature_extractor:250).
        """
        size = query_features.get("image_size") if query_features else None
        if size is None or len(size) < 2:
            return None
        return inlier_spread(pts_q, float(size[1]), float(size[0]))

    def _try_sift_rescue(
        self,
        rotated_frame: np.ndarray,
        rotated_mask: np.ndarray | None,
        candidates: list,
    ) -> tuple | None:
        """RESEARCH 2.2: одноразовий SIFT+LightGlue перезапуск матчингу.

        Повертає (candidate_id, H, inliers, mkpts_q_in, mkpts_r_in,
        total_matches, rmse) або None. Координати SIFT-точок — у тій самій
        системі rotated_frame, що й ALIKED, тож даунстрім-композиція гомографій
        не змінюється.
        """
        from src.localization.matcher import extract_sift_features

        try:
            q_sift = extract_sift_features(
                rotated_frame,
                rotated_mask,
                get_cfg(self.config, "database.sift_max_keypoints", 2048),
            )
        except Exception as e:
            logger.warning(f"SIFT rescue: query extraction failed: {e}")
            return None
        if len(q_sift["keypoints"]) < self.min_matches:
            return None

        best: tuple | None = None
        with Telemetry.profile("sift_rescue"):
            for cand_id, _score in candidates[: self._sift_fallback_max_cand]:
                try:
                    ref_sift = self.database.get_sift_features(cand_id)
                except (ValueError, KeyError):
                    continue
                mkq, mkr = self.matcher.match_sift(q_sift, ref_sift)
                if len(mkq) < self.min_matches:
                    continue
                H, mask = GeometryTransforms.estimate_homography(
                    mkq,
                    mkr,
                    ransac_threshold=self.ransac_thresh,
                    backend=self.homography_backend,
                    use_mad_ransac=self.use_mad_ransac,
                    mad_k_factor=self.mad_k_factor,
                )
                if H is None:
                    continue
                inl_mask = mask.ravel().astype(bool)
                inliers = int(np.sum(inl_mask))
                if inliers < self.min_matches:
                    continue
                pts_q_in, pts_r_in = mkq[inl_mask], mkr[inl_mask]
                proj = GeometryTransforms.apply_homography(pts_q_in, H)
                rmse = float(np.sqrt(np.mean(np.sum((proj - pts_r_in) ** 2, axis=1))))
                if best is None or inliers > best[2]:
                    best = (cand_id, H, inliers, pts_q_in, pts_r_in, len(mkq), rmse)

        if best is not None:
            logger.info(
                f"SIFT rescue SUCCEEDED: frame={best[0]}, inliers={best[2]}, "
                f"rmse={best[6]:.2f} (ALIKED had failed — likely in-plane rotation "
                f"or extreme oblique view)"
            )
        return best

    def _localize_by_reference_frame(self, frame_id: int, score: float) -> dict:
        return self._result_builder.fallback(frame_id, score, self.database, self.calibration)

    def _log_failure(self, error_type: str, inliers: int = 0, details: str = "") -> None:
        self._failure_logger.log(error_type, inliers, details)
