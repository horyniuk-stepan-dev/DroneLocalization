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

        # Multi-source database and calibration management
        self.db_manager = db_manager  # MultiDatabaseManager | None
        self.calib_manager = calib_manager  # MultiCalibrationManager | None
        self._active_source_id: str | None = None

        # Defaults synchronized with APP_CONFIG via get_cfg()
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
            zscore_enabled=get_cfg(self.config, "tracking.outlier_zscore_enabled", True),
            mahalanobis_enabled=get_cfg(self.config, "tracking.outlier_mahalanobis_enabled", False),
            chi2_threshold=get_cfg(self.config, "tracking.outlier_chi2_threshold", 13.816),
        )
        self._maha_gate_enabled = get_cfg(
            self.config, "tracking.outlier_mahalanobis_enabled", False
        )

        # Sliding Window Smoother for trajectory optimization over keyframe fixes & optical flow.
        self._smoother = None
        if get_cfg(self.config, "tracking.smoother_enabled", False):
            from src.tracking.smoother import SlidingWindowSmoother

            self._smoother = SlidingWindowSmoother(
                window=get_cfg(self.config, "tracking.smoother_window", 60),
                huber_k=get_cfg(self.config, "tracking.smoother_huber_k", 1.2),
                fix_sigma_base_m=get_cfg(self.config, "tracking.smoother_fix_sigma_base_m", 5.0),
                odom_sigma_base_m=get_cfg(self.config, "tracking.smoother_odom_sigma_base_m", 3.0),
                max_correction_m=get_cfg(self.config, "tracking.smoother_max_correction_m", 50.0),
                entry_prior_sigma_m=get_cfg(
                    self.config, "tracking.smoother_entry_prior_sigma_m", 15.0
                ),
                irls_iterations=get_cfg(self.config, "tracking.smoother_irls_iterations", 4),
                correction_lag=get_cfg(self.config, "tracking.smoother_correction_lag", 10),
                deadband_m=get_cfg(self.config, "tracking.smoother_deadband_m", 2.0),
                gain=get_cfg(self.config, "tracking.smoother_gain", 0.25),
                max_step_m=get_cfg(self.config, "tracking.smoother_max_step_m", 3.0),
            )

        # Retriever: single-database mode fallback
        self.retriever = None
        if self.db_manager is None:
            if hasattr(self.database, "lance_table") and self.database.lance_table is not None:
                self.retriever = LanceDBRetrieval(self.database.lance_table)
            else:
                self.retriever = FastRetrieval(self.database.global_descriptors)

        self.model_manager = self.config.get("_model_manager", None)
        self.fallback_enabled = get_cfg(self.config, "localization.enable_lightglue_fallback", True)
        self.min_inliers_for_accept = get_cfg(self.config, "localization.min_inliers_accept", 10)
        self.retrieval_top_k = get_cfg(self.config, "localization.retrieval_top_k", 8)
        # SIFT + LightGlue emergency fallback
        self._sift_fallback = get_cfg(self.config, "localization.sift_fallback", False)
        self._sift_fallback_max_cand = get_cfg(
            self.config, "localization.sift_fallback_max_candidates", 3
        )
        self.early_stop_inliers = get_cfg(self.config, "localization.early_stop_inliers", 30)

        # Temporal candidate prior for steady flight mode
        self._temporal_prior = get_cfg(self.config, "localization.temporal_candidate_prior", False)
        self._tp_window = int(get_cfg(self.config, "localization.temporal_prior_window", 2))
        self._tp_keep = int(get_cfg(self.config, "localization.temporal_prior_keep", 1))
        self._tp_min_mnn = int(get_cfg(self.config, "localization.temporal_prior_min_mnn", 20))
        self._tp_accept = int(
            get_cfg(self.config, "localization.temporal_prior_accept_inliers", 25)
        )
        self._tp_audit_every = int(
            get_cfg(self.config, "localization.temporal_prior_audit_every", 10)
        )
        self._tp_counter = 0
        self._tp_tries = 0
        self._tp_hits = 0

        # Monitoring of inlier spatial spread across frame
        self._spread_stats_enabled = get_cfg(
            self.config, "localization.spread_confidence_enabled", False
        )
        self._spread_log_every = get_cfg(self.config, "localization.spread_log_every", 50)
        self._spread_n = 0
        self._spread_n_low = 0
        self._spread_min = 1.0
        self._spread_sum = 0.0

        # Filtering anomalous shifts along optical flow track
        self._of_outlier_gate = get_cfg(self.config, "tracking.of_outlier_gate", False)
        # Distance correction considering real ground metric scale
        self._ground_scale_correction = get_cfg(
            self.config, "tracking.ground_scale_correction", False
        )
        # Calculating local frame-to-frame velocity instead of accumulated
        self._of_local_speed = get_cfg(self.config, "tracking.of_local_speed", False)
        self._last_of_raw: np.ndarray | None = None

        # Accepting measurements when strong independent geometric evidence is present
        self._trust_strong = get_cfg(self.config, "tracking.outlier_trust_strong_evidence", False)
        self._trust_min_inliers = int(
            get_cfg(self.config, "tracking.outlier_trust_min_inliers", 100)
        )
        self._trust_min_flow_q = float(
            get_cfg(self.config, "tracking.outlier_trust_min_flow_quality", 0.5)
        )

        # Fix #1: Guard against infinite loop when outside coverage bounds
        self._consecutive_failures = 0
        self._max_failures = get_cfg(self.config, "localization.max_consecutive_failures", 10)

        # Normalizing input frame resolution to DB reference resolution
        self.normalizer = ResolutionNormalizer(ref_frame_width, ref_frame_height)
        self._last_scale = 1.0

        # A3: temporal prior on rotation angle — angle of last successful
        # localization; full 4-angle scan only on score dip or failure
        self._last_best_angle: int | None = None

        # ── ScaleManager: GSD-ratio estimation for altitude-invariant localization ─
        self._scale_manager = ScaleManager(self.config)

        # Depth-based scale hint (soft pyramid reorder; hint only, never a hard scale).
        self._db_depth_scale = getattr(self.database, "median_depth_scale", None)
        self._use_depth_hint = get_cfg(self.config, "localization.scale_use_depth_hint", True)
        self._depth_hint_every_n = get_cfg(self.config, "localization.depth_hint_every_n", 30)
        self._depth_estimator = None
        self._depth_hint_counter = 0

        # ── Debug views: independent depth inference for window (separate cadence) ─
        self._debug_depth_every_n = get_cfg(self.config, "debug_views.depth_every_n_keyframes", 1)
        self._debug_depth_estimator = None
        self._debug_depth_counter = 0

        # ── Patchify: multi-patch retrieval ────────────────────────────────────
        # IMPORTANT: PatchifyRetrieval is initialized ONLY IF:
        #   1. Enabled via config
        #   2. Database contains patch_descriptors (i.e. built with use_patchify=True)
        # If either condition is not met — patchify is silently disabled (backward compat).
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
                    patchify_batch = get_cfg(self.config, "localization.patchify_batch_size", 1)
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
                logger.info("Patchify enabled in config but database has no patch_descriptors. ")

        self._candidate_retriever = CandidateRetriever(
            self.db_manager, self.retriever, self.patchify_retrieval, self.config
        )
        self._geometric_verifier = GeometricVerifier(
            self.matcher,
            self.min_matches,
            self.ransac_thresh,
            self.homography_backend,
            self.use_mad_ransac,
            self.mad_k_factor,
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
        """Last successful localization state (H, affine, angle, source_id) or None.

        Public access instead of reading private _last_state externally.
        """
        return getattr(self, "_last_state", None)

    def _sync_ground_scale(self, lat: float) -> None:
        """Updates projection-to-ground metric multiplier in outlier detector.

        Flag-gate: when tracking.ground_scale_correction is disabled multiplier
        remains 1.0 (legacy behavior). Latitude is taken from
        fresh fix; during mission cos(lat) changes by ~1e-5, so lag
        of one frame does not matter.
        """
        if not self._ground_scale_correction:
            return
        converter = getattr(self.calibration, "converter", None)
        if converter is None:
            return
        try:
            self.outlier_detector.set_ground_scale(converter.ground_scale_factor(lat))
        except Exception as e:  # noqa: BLE001 — correction must not crash localization
            logger.warning(f"Ground-scale sync failed ({type(e).__name__}: {e})")

    def reset_session(self) -> None:
        """Resets tracking session state (filters, counters, angle prior).

        Call at start of new tracking to avoid false
        predictions based on previous session.
        """
        self.trajectory_filter.reset()
        self.outlier_detector.reset()
        self._last_of_raw = None
        self._consecutive_failures = 0
        self._last_best_angle = None
        self._last_state = None
        self._scale_manager.reset()
        self._debug_depth_counter = 0
        self._tp_counter = 0
        self._tp_tries = 0
        self._tp_hits = 0
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
        """Debug: independent depth inference for window (separate cadence).

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
            # relative scale from center (like get_relative_scale, without 2nd inference)
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
        # Fix #1: If too many consecutive failures occurred — return out_of_coverage
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

        # Normalization to database reference resolution
        query_frame, self._last_scale = self.normalizer.normalize(query_frame)
        if static_mask is not None:
            static_mask = self.normalizer.normalize_mask(static_mask)
        height, width = query_frame.shape[:2]

        # Depth hint: soft reorder of the scale pyramid toward the DB GSD (every N keyframes).
        self._maybe_set_depth_hint(query_frame)
        # Debug: depth map for window (independent of localization success).
        self._maybe_collect_depth(query_frame, collector)

        angles_to_try = [0, 90, 180, 270] if self.enable_auto_rotation else [0]

        top_k = self.retrieval_top_k

        # §A1: cache (angle, scale) -> (frame, mask, crop, features) for ONE call.
        # If temporal hypothesis failed, and full path chose the same
        # angle and scale — ALIKED is not recomputed (217-339 ms on GTX 1650).
        _feat_cache: dict = {}

        # ── §A1: attempt to localize WITHOUT global descriptor ──────────
        # yaw_hint_deg disables this path: external heading is new information
        # about orientation, it must be processed by full rotation path.
        _tp = None
        if self._temporal_prior and yaw_hint_deg is None:
            self._tp_counter += 1
            audit = self._tp_audit_every
            if audit <= 0 or (self._tp_counter % audit) != 0:
                self._tp_tries += 1
                _tp = self._try_temporal_prior(query_frame, static_mask, _feat_cache)
                if _tp is not None:
                    self._tp_hits += 1
                if self._tp_tries % 50 == 0:
                    logger.info(
                        f"[temporal-prior] tries={self._tp_tries} "
                        f"hits={self._tp_hits} "
                        f"({100.0 * self._tp_hits / self._tp_tries:.0f}%)"
                    )

        if _tp is not None:
            (
                ver,
                best_global_angle,
                best_scale,
                best_rotated_frame,
                best_rotated_mask,
                _crop_info,
                best_query_features,
                best_global_candidates,
            ) = _tp
            # Retrieval was not executed — global score does not exist. -1.0
            # deliberately fails retrieval_only_min_score, so fallback
            # 'by similarity' on this path will not trigger; candidates are already
            # selected by presence of propagated calibration.
            best_global_score = -1.0
            best_source_id_per_angle = self._active_source_id
            if collector is not None:
                collector.global_score = best_global_score
                collector.global_angle = int(best_global_angle)
                collector.scale = float(best_scale)
                collector.retrieval_candidates = [
                    (int(cid), float(sc)) for cid, sc in best_global_candidates
                ]
                collector.rotated_frame = best_rotated_frame
                collector.query_features = best_query_features
            logger.debug(
                f"Temporal prior HIT: frame={ver.candidate_id}, "
                f"inliers={ver.inliers}, angle={best_global_angle} deg "
                f"(global descriptor skipped)"
            )
        else:
            # ── RESEARCH 2.3: external yaw-hint (simulator / telemetry) ────────
            # yaw_hint_deg — CW angle in degrees to rotate the frame to match
            # DB orientation (north-up); conversion from drone heading
            # is done by caller. Quantized to 90 degrees — full rotation path operates with
            # k*90. False hint self-heals: if retrieval-score of prior-angle
            # is lower than rotation_rescan_min_score, RotationSelector performs
            # full batched 4-angle scan.
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

            # ── Step 1.5a: Switching database/calibration for multi-mode ───
            if self.db_manager is not None and best_source_id_per_angle is not None:
                self._active_source_id = best_source_id_per_angle
                self.database = self.db_manager.get_database(best_source_id_per_angle)
                if self.calib_manager is not None:
                    self.calibration = self.calib_manager.get(best_source_id_per_angle)
                logger.debug(f"Active source switched to '{best_source_id_per_angle}'")

            # ── Steps 1.5 + 1.5b + 2: rotation, GSD normalization, ALIKED ─────────
            # §A1: via _prepare_and_extract, so features already computed by failed
            # temporal hypothesis at the same (angle, scale) are not computed
            # a second time. Step 1.6 (patchify-expand) moved BELOW extraction — they are
            # independent: expand reads only the frame, not features.
            (
                best_rotated_frame,
                best_rotated_mask,
                _crop_info,
                best_query_features,
            ) = self._prepare_and_extract(
                query_frame,
                static_mask,
                best_global_angle,
                best_scale,
                _feat_cache,
                # §2.2: selector has already rotated and scaled this exact frame
                prepared=(rot.frame, rot.crop_info),
            )

            # ── Step 1.6: Patchify candidate expansion (only for best angle) ─
            # Run ONCE after angle selection — not in a loop.
            # Patchify adds candidates that DINOv2 CLS-token might have missed
            # (e.g., during altitude change).
            best_global_candidates = self._candidate_retriever.expand(
                best_rotated_frame, best_global_candidates, top_k
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

        # ADDENDUM 1.1: inlier spread — computed BEFORE SIFT fallback and
        # recomputed after it, because it replaces the point set.
        best_spread = self._inlier_spread(best_mkpts_q_inliers, best_query_features)
        if collector is not None:
            collector.spread = best_spread

        # ── RESEARCH 2.2: emergency SIFT+LightGlue fallback ────────────────────
        # ALIKED (like SuperPoint) loses matches under large in-plane rotation
        # and extreme tilt [ISPRS 2025; MDPI RS 17(22)]. Single
        # re-run via rotation-invariant SIFT + LightGlue(sift) saves
        # the frame before it goes into retrieval-only fallback.
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
                # Points replaced by SIFT — spread is no longer what was above.
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

        # ── Step 4: Obtaining candidate affine matrix ─────────────────────
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

        # Dimensions of rotated normalized image
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

        # ── FOV-remap (IMPLEMENTATION_PLAN, Phase 1.2) ────────────────────────────────────
        # H found in GSD-normalized frame coordinates (crop/resize).
        # Composition with A (rotated->normalized) transforms H into coordinates
        # of rotated frame — further center (Step 6), FOV (Step 8), OF-state
        # (Step 5) and scale-prior (update_from_homography) are computed in single
        # coordinate system. Without this, for r < 0.85 center is shifted by
        # ~(1-r)/2 frame, polygon inflated by 1/r, and prior collapses to 1.
        if _crop_info is not None and _crop_info.resize_scale != 1.0:
            n_h, n_w = best_rotated_frame.shape[:2]
            _A_norm = crop_to_affine(_crop_info, n_w, n_h)
            M_query_to_ref = M_query_to_ref @ _A_norm
            if best_mkpts_q_inliers is not None and len(best_mkpts_q_inliers) > 0:
                # mkpts remain in normalized coordinates only for
                # collector (draws on normalized frame); for
                # build_fov (clamped to rot_width/rot_height) converted to
                # rotated frame coordinates.
                _A_inv = crop_to_affine(_crop_info, n_w, n_h, inverse=True)
                best_mkpts_q_inliers = GeometryTransforms.apply_homography(
                    np.asarray(best_mkpts_q_inliers, dtype=np.float64), _A_inv
                )

        # ── Step 5: State for Optical Flow (commit — AFTER outlier gate) ─────
        pending_state = {
            "H": M_query_to_ref,
            "affine": affine_ref,
            "candidate_id": best_candidate_id,
            "inliers": best_inliers,
            "global_angle": best_global_angle,
            "source_id": self._active_source_id,
            # Normalization scale of THIS SPECIFIC keyframe: OF operates in
            # frame system belonging to H (fresh self._last_scale on
            # subsequent frames may already differ).
            "scale": self._last_scale,
        }

        # ── Step 6: Query center → Reference → Metric → GPS ─────────────────
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

        # ── Step 7: Outlier filtering ─────────────────────────────────────
        # Strong geometry beats kinematic prior: fix supported by hundreds of
        # RANSAC inliers should not be discarded due to platform speed assumptions.
        # Position is still appended to history so detector window
        # corresponded to reality, not to a filtered version of it.
        _strong = self._trust_strong and best_inliers >= self._trust_min_inliers
        if _strong:
            logger.debug(
                f"Kinematic gate bypassed: {best_inliers} inliers "
                f">= {self._trust_min_inliers} (geometry outranks motion prior)"
            )
        # Mahalanobis-gate (flag): d^2 computed BEFORE filter update, using pure
        # function. noise_scale=1.0 — base R: confidence at this point in code is not
        # yet computed (it is below in Step 8), and reordering for the sake of
        # the gate would mean changing more than the task requires. Consequence: for
        # fixes with low confidence gate is slightly stricter than subsequent update.
        _maha_d2 = (
            self.trajectory_filter.mahalanobis_sq(metric_pt, dt=dt, noise_scale=1.0)
            if self._maha_gate_enabled
            else None
        )
        if not _strong and self.outlier_detector.is_outlier(metric_pt, dt, maha_d2=_maha_d2):
            logger.warning(
                f"Outlier filtered | matched_frame={best_candidate_id}, "
                f"metric=({mx:.1f}, {my:.1f}), inliers={best_inliers}, dt={dt:.3f}s. "
                f"Position jump was too large relative to recent trajectory."
            )
            self._log_failure(FAILURE_TYPES["Outlier detected"], inliers=best_inliers)
            # RESEARCH 3.1: rejected fix still enters the window
            # of the smoother — Huber weight arbitrates instead of binary rejection
            # (insures Z-score false positives during sharp maneuvers).
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

        # BUGFIX (OF-seam): state commit ONLY AFTER outlier gate. Previously state
        # was committed at Step 5 — both for rejected frames and prior to
        # homography-failure return — so OF received H inconsistent with
        # worker prev_pts (does not rebase points without success).
        self._last_state = pending_state
        self._consecutive_failures = 0

        # Confidence computed BEFORE filtering — B2: adaptive measurement noise,
        # weak localization affects trajectory less, confident — more
        confidence = self._compute_confidence(
            best_candidate_id, best_inliers, best_total_matches, best_rmse, best_spread
        )

        filtered_pt = self.trajectory_filter.update(
            metric_pt, dt=dt, noise_scale=1.0 / max(confidence, 0.25)
        )
        # RESEARCH 3.1: back-end smoother — fix window + OF-odometry;
        # KF correction by shift BEFORE writing to detector history and GPS/FOV,
        # so correction lands in THIS frame.
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
                logger.debug(f"Smoother correction applied: ({corr[0]:+.2f}, {corr[1]:+.2f}) m")
        self.outlier_detector.add_position(filtered_pt, dt=dt)
        # New keyframe restarts LK, so chain of local OF comparisons
        # breaks: first OF after keyframe should measure shift FROM keyframe
        # (ref=None -> database = newly added position in window), not from OF-measurement
        # of previous cycle. Otherwise database of shift and dt diverge again.
        self._last_of_raw = None
        lat, lon = self.calibration.converter.metric_to_gps(
            float(filtered_pt[0]), float(filtered_pt[1])
        )
        self._sync_ground_scale(lat)
        dx, dy = filtered_pt[0] - metric_pt[0], filtered_pt[1] - metric_pt[1]

        # -- Step 8: FOV calculation -------------------------------------------
        gps_corners = self._result_builder.build_fov(
            M_query_to_ref,
            affine_ref,
            rot_width,
            rot_height,
            best_mkpts_q_inliers,
            self.calibration.converter,
            dx,
            dy,
            mx,
            my,
            filtered_pt,
            best_candidate_id,
        )

        logger.debug(
            f"Localize Frame {best_candidate_id}: Center transformed via Homography (8 DoF)"
        )
        logger.debug(f"Sample Center METRIC: ({mx:.1f}, {my:.1f})")
        source_str = f" | source={self._active_source_id}" if self._active_source_id else ""
        logger.success(
            f"Localized ({lat:.6f}, {lon:.6f}) | frame={best_candidate_id}{source_str} | "
            f"metric=({mx:.1f}, {my:.1f}) | inliers={best_inliers} | conf={confidence:.2f}"
        )

        # A3: remember angle for temporal prior of next keyframe
        self._last_best_angle = best_global_angle

        # Scale prior: extract scale from H for the next keyframe
        self._scale_manager.update_from_homography(M_query_to_ref, rot_width, rot_height)

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
        """Localization based on pixel shift from Optical Flow.

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

        # Restoring database/calibration for saved source_id (multi-mode)
        last_source_id = self._last_state.get("source_id")
        if last_source_id is not None and self.db_manager is not None:
            self.database = self.db_manager.get_database(last_source_id)
            if self.calib_manager is not None:
                self.calibration = self.calib_manager.get(last_source_id)

        # Scale from saved keyframe state (consistent with its H);
        # fallback to _last_scale for states recorded prior to this field.
        scale = self._last_state.get("scale", self._last_scale)
        angle = self._last_state.get("global_angle", 0)

        # -- 1. Shift vector: original space -> normalized + rotated --
        # Scaling to normalized space
        sdx = dx_px * scale
        sdy = dy_px * scale

        # Rotating shift vector according to frame orientation.
        # H constructed in rotated normalized frame space, so shift
        # must also be in the same coordinate system.
        a, b, c, d = _ROTATION_VEC.get(angle, (1, 0, 0, 1))
        rot_sdx = a * sdx + b * sdy
        rot_sdy = c * sdx + d * sdy

        # -- 2. Frame dimensions: original -> normalized + rotated --------
        if angle in (90, 270):
            # 90 deg / 270 deg: rows and columns swap
            norm_rot_w = rot_height * scale
            norm_rot_h = rot_width * scale
        else:
            norm_rot_w = rot_width * scale
            norm_rot_h = rot_height * scale

        # -- 3. Current frame center in previous coordinate system --------
        center_query_shifted = None

        if flow_affine is not None:
            # B4: full similarity S (original px, KF->current). Point in KF-frame,
            # currently located at center: p0 = S^-1 @ center. Then p0 is mapped
            # normalized -> rotated (same transform as for frame).
            try:
                S3 = np.vstack([np.asarray(flow_affine, dtype=np.float64), [0.0, 0.0, 1.0]])
                S_inv = np.linalg.inv(S3)
                cx0, cy0 = rot_width / 2.0, rot_height / 2.0
                p0x = S_inv[0, 0] * cx0 + S_inv[0, 1] * cy0 + S_inv[0, 2]
                p0y = S_inv[1, 0] * cx0 + S_inv[1, 1] * cy0 + S_inv[1, 2]
                # original → normalized
                p0x *= scale
                p0y *= scale
                # normalized -> rotated (point mapping via np.rot90, verified)
                w_n, h_n = rot_width * scale, rot_height * scale
                rx, ry = _rotate_point_np90(p0x, p0y, w_n, h_n, angle)
                center_query_shifted = np.array([[rx, ry]], dtype=np.float64)
            except np.linalg.LinAlgError:
                center_query_shifted = None  # вироджена S → fallback на трансляцію

        if center_query_shifted is None:
            # Fallback: pure translation — if since KF points shifted by
            # (dx, dy), center corresponds to point (center - displacement) in KF frame.
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

        # -- Outlier gate on OF path (audit item 1.2), flag-gated ------------
        # Structural gap: keyframe path checks is_outlier (Step 7), while OF —
        # did not, it only appended position to history. At keyframe_interval=30 this is
        # 29 out of 30 positions going outward without any check: loss
        # tracking LK onto cloud or water surface went straight into GPS.
        # Default False = legacy behavior bitwise.
        # High flow_quality is independent evidence that flow is consistent: shift
        # real, not tracking loss. Measured on live run: real
        # LK slips gave 0.017-0.035, while falsely rejected fast motions —
        # 0.625-1.0. Kinematic gate does not distinguish them, but this threshold does.
        _strong_flow = (
            self._trust_strong
            and flow_quality is not None
            and float(flow_quality) >= self._trust_min_flow_q
        )
        if _strong_flow:
            logger.debug(
                f"OF kinematic gate bypassed: flow_quality={float(flow_quality):.3f} "
                f">= {self._trust_min_flow_q} (flow is self-consistent)"
            )
        # Instantaneous velocity reference point: previous RAW OF measurement (even
        # if rejected). Without it reference is last accepted position, usually
        # keyframe, and speed accumulates along with LK shift.
        _of_ref = self._last_of_raw if self._of_local_speed else None
        # noise_scale=1.5 matches base multiplier of OF-branch update() at
        # of_conf=1.0 — OF measurement is inherently noisier than keyframe-fix.
        _maha_d2 = (
            self.trajectory_filter.mahalanobis_sq(metric_pt, dt=dt, noise_scale=1.5)
            if self._maha_gate_enabled
            else None
        )
        _is_out = (
            self._of_outlier_gate
            and not _strong_flow
            and self.outlier_detector.is_outlier(
                metric_pt, dt, ref_position=_of_ref, maha_d2=_maha_d2
            )
        )
        # Updating BEFORE early return: next frame must be compared with this one
        # measurement regardless of whether we accepted it or not.
        self._last_of_raw = metric_pt.copy()
        if _is_out:
            logger.warning(
                f"OF outlier filtered | metric=({mx:.1f}, {my:.1f}), dt={dt:.3f}s, "
                f"flow_quality={flow_quality if flow_quality is None else round(flow_quality, 3)}. "
                f"Optical flow likely lost lock (clouds, water, motion blur)."
            )
            self._log_failure(FAILURE_TYPES["Outlier detected"])
            # Rejected OF still goes into smoother window as odometry:
            # Huber weight arbitrates better than binary rejection (same logic,
            # as for rejected keyframes).
            if self._smoother is not None:
                self._smoother.note_of(metric_pt, dt=dt, quality=flow_quality)
            return {"success": False, "error": "OF outlier — position jump filtered"}

        # RESEARCH 3.1: raw OF fix into smoother window — relative odometry,
        # tied to H of last accepted keyframe.
        if self._smoother is not None:
            self._smoother.note_of(metric_pt, dt=dt, quality=flow_quality)

        # B2: honest OF confidence (previously hardcoded 0.8) + higher measurement noise
        # for Kalman (OF is relative measurement, drifts from KF)
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
        self._sync_ground_scale(lat)

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

    # ── PIPELINE_OPTIMIZATION_PLAN §A1 ──────────────────────────────────────

    def _prepare_and_extract(
        self,
        query_frame: np.ndarray,
        static_mask: np.ndarray | None,
        angle: int,
        scale: float,
        cache: dict,
        prepared: tuple | None = None,
    ) -> tuple:
        """Rotate frame by ``angle``, normalize to ``scale``, extract ALIKED.

        ``cache`` живе рівно один виклик ``localize_frame``: якщо темпоральна
        гіпотеза провалилась і повний шлях обрав ті самі (кут, масштаб),
        екстракція не повторюється. Повертає
        ``(rotated_frame, rotated_mask, crop_info, features)``.

        ``prepared`` (аудит §2.2) — ``(frame, crop_info)`` від RotationSelector
        для ТІЄЇ САМОЇ пари (кут, масштаб): він уже зробив rot90 і GSD-resize,
        щоб порахувати глобальний дескриптор. Тоді тут лишається тільки маска
        і ALIKED — на 1080p це мінус ~6 МБ memcpy і один resize на keyframe.
        Маску все одно доводиться готувати окремо: селектор її не бачить.
        """
        key = (int(angle), round(float(scale), 3))
        cached = cache.get(key)
        if cached is not None:
            return cached

        k = int(angle) // 90
        rot_mask = np.rot90(static_mask, k=k).copy() if static_mask is not None else None
        needs_gsd = abs(float(scale) - 1.0) > 0.15

        if prepared is not None and prepared[0] is not None:
            # Frame is already prepared by selector for these same (angle, scale).
            rotated, crop_info = prepared
            if needs_gsd and rot_mask is not None:
                rot_mask, _ = self._scale_manager.normalize(rot_mask, float(scale))
        else:
            rotated = np.rot90(query_frame, k=k).copy()
            crop_info = None
            # GSD-normalization: in steady flight scale ≈ 1.0 and this is a no-op.
            if needs_gsd:
                rotated, crop_info = self._scale_manager.normalize(rotated, float(scale))
                if rot_mask is not None:
                    rot_mask, _ = self._scale_manager.normalize(rot_mask, float(scale))
                logger.debug(
                    f"GSD-normalized frame for scale {scale:.2f}: "
                    f"{rotated.shape[1]}x{rotated.shape[0]}"
                )

        feats = self.feature_extractor.extract_local_features(rotated, static_mask=rot_mask)
        cache[key] = (rotated, rot_mask, crop_info, feats)
        return cache[key]

    def _tp_neighbour_ids(self) -> list[int]:
        """Candidates from neighborhood of last match — without any forward-pass.

        Порядок: сам останній кадр, далі симетрично id±1, id±2 … Кадри без
        пропагованої калібрації відкидаються одразу: без ``frame_affine``
        локалізація по них однаково не завершиться, а перевірка — це
        звернення до масиву.
        """
        st = self.last_state
        if not st:
            return []
        cid = int(st.get("candidate_id", -1))
        if cid < 0:
            return []
        ids: list[int] = []
        for d in range(0, max(0, self._tp_window) + 1):
            for c in (cid,) if d == 0 else (cid - d, cid + d):
                if c < 0 or c in ids:
                    continue
                try:
                    if self.database.get_frame_affine(c) is None:
                        continue
                except Exception as e:  # noqa: BLE001 — БД може не мати кадру
                    logger.debug(f"Temporal prior: frame {c} rejected ({e})")
                    continue
                ids.append(c)
        return ids

    def _try_temporal_prior(
        self, query_frame: np.ndarray, static_mask: np.ndarray | None, cache: dict
    ) -> tuple | None:
        """Item A1: localization without global descriptor.

        Повертає кортеж для гілки в ``localize_frame`` або ``None`` — тоді
        викликач іде повним шляхом (фічі вже лежать у ``cache``, тож ALIKED
        не повториться).

        Ціна промаху навмисно тримається низькою: гіпотеза спершу перевіряється
        MNN-скорером (один матмул на кандидата), і лише пройшовши поріг
        ``temporal_prior_min_mnn``, доходить до LightGlue.
        """
        angle = self._last_best_angle
        if angle is None:
            return None
        ids = self._tp_neighbour_ids()
        if not ids:
            return None

        scale = self._scale_manager.prior
        scale = 1.0 if scale is None else float(scale)

        rotated, rot_mask, crop_info, feats = self._prepare_and_extract(
            query_frame, static_mask, angle, scale, cache
        )

        cands = [(int(i), 0.0) for i in ids]
        ref_cache: dict = {}
        scored = self._geometric_verifier.mnn_counts(feats, cands, self.database, ref_cache)
        if not scored:
            return None
        scored.sort(key=lambda t: -t[0])
        if scored[0][0] < self._tp_min_mnn:
            logger.debug(
                f"Temporal prior: MNN probe too weak "
                f"({scored[0][0]} < {self._tp_min_mnn}) — full path"
            )
            return None

        probe = [(cid, float(m)) for m, cid, _ in scored[: max(1, self._tp_keep)]]
        ver = self._geometric_verifier.verify(feats, probe, self.database, ref_cache=ref_cache)
        if ver is None or ver.inliers < self._tp_accept:
            got = ver.inliers if ver is not None else 0
            logger.debug(
                f"Temporal prior: rejected ({got} inliers < {self._tp_accept}) — full path"
            )
            return None

        return (ver, int(angle), float(scale), rotated, rot_mask, crop_info, feats, probe)

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
        """Accumulates spread statistics and periodically logs them.

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
        """ADDENDUM 1.1: inlier spread in query frame coordinate system.

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
        """RESEARCH 2.2: single-attempt SIFT+LightGlue matching rerun.

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
