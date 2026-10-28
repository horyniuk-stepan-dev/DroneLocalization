import numpy as np

from config import get_cfg
from src.geometry.frame_transform import project_frame_points
from src.geometry.point_spread import inlier_spread
from src.geometry.transformations import GeometryTransforms
from src.localization.altitude_hint import AltitudeHint
from src.localization.candidate_retriever import CandidateRetriever
from src.localization.failure_log import FAILURE_TYPES, FailureLogger
from src.localization.geometric_verifier import GeometricVerifier
from src.localization.keypoint_budget import top_keypoints
from src.localization.layer_search import LayerSearch
from src.localization.matcher import FastRetrieval, LanceDBRetrieval
from src.localization.query_rotation import (
    angle_distance_deg,
    is_continuous,
    next_prior_deg,
    norm_deg,
    rotation_matrix,
    scan_angles,
    view_mask,
)
from src.localization.result_builder import ResultBuilder
from src.localization.rotation_geometry import _ROTATION_VEC, _rotate_point_np90
from src.localization.rotation_selector import RotationSelector
from src.localization.scale_manager import ScaleManager, crop_to_affine
from src.localization.spatial_prior import SpatialPrior
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
        # Independent relative maps have independent gauges. Search only the
        # selected database; never compare or switch them by geographic distance.
        self.relative_mode = getattr(getattr(calibration, "converter", None), "mode", "") == "LOCAL"
        if self.relative_mode:
            db_manager = calib_manager = None
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
        self.ransac_max_iters = int(get_cfg(self.config, "homography.max_iters", 2000))
        self.ransac_confidence = float(get_cfg(self.config, "homography.confidence", 0.99))

        self.trajectory_filter = TrajectoryFilter(
            process_noise=get_cfg(self.config, "tracking.kalman_process_noise", 2.0),
            measurement_noise=get_cfg(self.config, "tracking.kalman_measurement_noise", 5.0),
            dt=1.0,
            reanchor_velocity_from_fixes=get_cfg(
                self.config, "tracking.kalman_reanchor_velocity_from_fixes", False
            ),
            max_gap_s=get_cfg(self.config, "tracking.kalman_max_gap_s", 0.0),
        )
        self.outlier_detector = OutlierDetector(
            window_size=get_cfg(self.config, "tracking.outlier_window", 10),
            threshold_std=get_cfg(self.config, "tracking.outlier_threshold_std", 4.0),
            max_speed_mps=get_cfg(self.config, "tracking.max_speed_mps", 120.0),
            max_consecutive=get_cfg(self.config, "tracking.max_consecutive_outliers", 5),
            zscore_enabled=get_cfg(self.config, "tracking.outlier_zscore_enabled", True),
            mahalanobis_enabled=get_cfg(self.config, "tracking.outlier_mahalanobis_enabled", False),
            chi2_threshold=get_cfg(self.config, "tracking.outlier_chi2_threshold", 13.816),
            reacquire_consistent=get_cfg(
                self.config, "tracking.outlier_reacquire_consistent", False
            ),
            reacquire_tol_m=get_cfg(self.config, "tracking.outlier_reacquire_tol_m", 15.0),
            reacquire_max_wait=get_cfg(self.config, "tracking.outlier_reacquire_max_wait", 10),
        )
        self._maha_gate_enabled = get_cfg(
            self.config, "tracking.outlier_mahalanobis_enabled", False
        )

        # Sliding Window Smoother for trajectory optimization over keyframe fixes & optical flow.
        self._smoother = None
        if not self.relative_mode and get_cfg(self.config, "tracking.smoother_enabled", False):
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
                odom_huber_k=get_cfg(self.config, "tracking.smoother_odom_huber_k", 0.0),
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
        # Temporal prior while a yaw hint is present (flag; None = legacy:
        # any hint disables the prior). Measured 2026-10-28 on route3min
        # variants: with telemetry the prior never ran, every keyframe paid a
        # full retrieval, and on other-epoch imagery (wayback 2019) keyframe
        # success fell 0.87 -> 0.15. A hint that agrees with the last fix's
        # angle within this many degrees carries no new orientation info.
        _tol = get_cfg(self.config, "localization.temporal_prior_yaw_hint_tol_deg", None)
        self._tp_hint_tol = None if _tol is None else float(_tol)
        # Strongest-N query keypoints for the temporal-prior check (0 = all).
        self._tp_max_kp = int(get_cfg(self.config, "localization.temporal_prior_max_keypoints", 0))
        # ...only after a fix with at least this many inliers (0 = always); see
        # _tp_keypoint_budget.
        self._tp_kp_min_inliers = int(
            get_cfg(self.config, "localization.temporal_prior_kp_min_inliers", 0)
        )
        self._prev_fix_inliers = 0
        self._tp_counter = 0
        self._tp_tries = 0
        self._tp_hits = 0
        # Spatial prior: second temporal-prior stage near the dead-reckoned
        # position (flag localization.spatial_prior, off by default).
        self._spatial_prior = SpatialPrior.from_config(self.config, enabled=not self.relative_mode)
        self._sp_now: float | None = None
        self._sp_frame_done = False
        self._tp_verified: list[int] = []  # ids the index stage verified this call
        self._sp_tries = 0
        self._sp_hits = 0

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
        # OF decorrelation (flag; 0.0 = legacy). Every OF position is projected
        # through the SAME keyframe homography (LK always tracks from the
        # keyframe), so the keyframe's error is shared by all of them; feeding
        # them to the KF as independent fixes counts that error N times. The OF
        # measurement noise is therefore inflated by (1 + growth * n), n = OF
        # samples since the keyframe, which caps their joint information.
        self._of_noise_growth = max(
            float(get_cfg(self.config, "tracking.of_noise_growth_per_step", 0.0)), 0.0
        )
        self._of_steps_since_kf = 0
        # KF prediction interval (flag; False = legacy). The worker's dt is
        # path-specific: a keyframe gets the time since the previous KEYFRAME
        # although the KF was last updated by optical flow ~1/6 s earlier, so the
        # constant-velocity model predicted ~0.9 s instead of ~0.17 s on every
        # keyframe and lagged in between (measured 2026-10-28: raw OF step ratio
        # 0.99, filtered 0.62-0.71; offline KF replay: jumps >10 m 190 -> 6).
        # True = KF dt from frame timestamps since its last update; outlier
        # detector and smoother keep the worker dt.
        # OF quality floor (flag; 0.0 = legacy). A lost LK track reports
        # flow_quality 0.026-0.042 (route3min variants 2026-10-28: all 11 OF
        # samples > 20 m off), good tracking >= 0.075 at its 0.5th percentile;
        # such samples slipped past the kinematic gate and dragged the KF up
        # to 178 m off for ~1 s.
        self._of_min_quality = max(
            float(get_cfg(self.config, "tracking.of_min_flow_quality", 0.0)), 0.0
        )
        # Base KF noise multiplier of an OF measurement (legacy 1.5, "OF is
        # noisier than a keyframe fix"). With the deeper LK pyramid and the
        # quality floor OF is as accurate as a fix; offline on six route3min
        # variants (2026-10-28) the KF output beat raw only when OF was trusted:
        # q 8 / 1.5 -> p95 5.9 m, max 50 m, 46 jumps; q 128 / 0.1 -> 4.1 m, 28 m, 5.
        self._of_noise_base = max(float(get_cfg(self.config, "tracking.of_noise_scale", 1.5)), 0.01)
        self._kf_dt_from_ts = bool(
            get_cfg(self.config, "tracking.kalman_dt_from_timestamps", False)
        )
        self._frame_timestamp: float | None = None

        # Accepting measurements when strong independent geometric evidence is present
        self._trust_strong = get_cfg(self.config, "tracking.outlier_trust_strong_evidence", False)
        self._trust_min_inliers = int(
            get_cfg(self.config, "tracking.outlier_trust_min_inliers", 100)
        )
        self._trust_min_flow_q = float(
            get_cfg(self.config, "tracking.outlier_trust_min_flow_quality", 0.5)
        )
        self._trusted_fix_max_filter_offset = float(
            get_cfg(self.config, "tracking.trusted_fix_max_filter_offset_m", 5.0)
        )

        # Fix #1: Guard against infinite loop when outside coverage bounds
        self._consecutive_failures = 0
        self._max_failures = get_cfg(self.config, "localization.max_consecutive_failures", 10)

        # Normalizing input frame resolution to DB reference resolution
        self.normalizer = ResolutionNormalizer(ref_frame_width, ref_frame_height)
        self._last_scale = 1.0
        self._last_state = None

        # A3: temporal prior on rotation angle — angle of last successful
        # localization; full 4-angle scan only on score dip or failure
        self._last_best_angle: float | None = None

        # localization.rotation_mode (src/localization/query_rotation.py)
        self._rot_continuous = is_continuous(self.config)
        self._rot_scan = scan_angles(
            get_cfg(self.config, "localization.rotation_scan_step_deg", 45.0)
        )
        self._rot_refine_min = float(
            get_cfg(self.config, "localization.rotation_refine_min_deg", 12.0)
        )
        self._rot_erode = int(get_cfg(self.config, "localization.rotation_border_erode_px", 8))
        if self._rot_continuous:
            logger.info(
                f"Rotation mode: continuous (scan {self._rot_scan}, "
                f"refine > {self._rot_refine_min:.0f} deg)"
            )

        # ── ScaleManager: GSD-ratio estimation for altitude-invariant localization ─
        self._scale_manager = ScaleManager(self.config)
        self._layer_search = LayerSearch(self.config)
        self._layer_clock = None

        # localization.altitude_hint: the expected scales of the layers nearest the
        # altitude (telemetry, else the last fix's scale) are tried first.
        _project_manager = self.config.get("_project_manager", None)
        self._altitude_hint = AltitudeHint.from_config(
            self.config, getattr(_project_manager, "settings", None)
        )
        self._telemetry_altitude_m = None
        self._altitude_norm_width = None
        self._altitude_hint_logged: set = set()

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
        # Multi-layer: one patch index per layer, built on the layer's first use. Frame ids
        # are per database, so patch hits may only be merged into candidates of their own
        # layer (one index from the first layer used to be merged into every layer's list).
        self.patchify_retrieval = None
        patchify_factory = None
        # localization.patchify_main_rank_margin = N > 0 (multi-layer): candidates found
        # only by patchify are verified after the main layer's standard candidates unless
        # they rank more than N places above them (src/database/source_priority.py).
        self._patchify_main_margin = int(
            get_cfg(self.config, "localization.patchify_main_rank_margin", 0) or 0
        )
        # localization.joint_candidate_ranking (multi-layer): one verification list from
        # the standard and patchify candidates of every layer, ordered by rank, layer
        # priority and patchify penalty (src/database/source_priority.py::joint_order).
        self._joint_ranking = bool(
            get_cfg(self.config, "localization.joint_candidate_ranking", False)
        )
        self._joint_top_k = int(get_cfg(self.config, "localization.joint_top_k", 5) or 5)
        self._joint_patch_penalty = int(
            get_cfg(self.config, "localization.joint_patchify_rank_penalty", 2) or 0
        )
        self._joint_rank_step = int(
            get_cfg(self.config, "localization.source_priority_rank_margin", 0) or 0
        )
        # localization.layer_fineness_weight: layer map accuracy (ground pixel size) joins
        # the project priority; localization.patchify_adaptive: patchify only where no real
        # layer is close in scale to the query (src/database/source_priority.py).
        self._fineness_weight = float(
            get_cfg(self.config, "localization.layer_fineness_weight", 0.0) or 0.0
        )
        self._patchify_adaptive = bool(
            get_cfg(self.config, "localization.patchify_adaptive", False)
        )
        self._patchify_gap_ratio = float(
            get_cfg(self.config, "localization.patchify_gap_ratio", 1.5) or 1.5
        )
        # localization.patchify_lazy: patch search only after the standard candidates fail.
        self._patchify_lazy = bool(get_cfg(self.config, "localization.patchify_lazy", False))
        self._layer_gsd_cache: dict = {}
        self._frame_query_gsd = None
        if get_cfg(self.config, "localization.use_patchify", False):
            if self.db_manager is not None:
                patchify_factory = self._build_patchify_retrieval
            else:
                self.patchify_retrieval = self._build_patchify_retrieval(self.database)

        self._candidate_retriever = CandidateRetriever(
            self.db_manager,
            self.retriever,
            self.patchify_retrieval,
            self.config,
            patchify_factory=patchify_factory,
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
            max_rmse_px=get_cfg(self.config, "localization.max_geometric_rmse_px", 4.0),
            min_inlier_ratio=get_cfg(self.config, "localization.geometric_min_inlier_ratio", 0.2),
            max_center_extrapolation=get_cfg(
                self.config, "localization.geometric_max_center_extrapolation", 0.1
            ),
            min_reference_eigenvalue=get_cfg(
                self.config, "localization.geometric_min_reference_eigenvalue", 1e-4
            ),
            ransac_max_iters=self.ransac_max_iters,
            ransac_confidence=self.ransac_confidence,
            prefilter_fallback_keep=get_cfg(self.config, "localization.prefilter_fallback_keep", 0),
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

    def _main_source_id(self):
        """Active layer with the highest priority (lowest ``priority``; main = 0).

        With localization.layer_fineness_weight > 0 the effective priority (project
        priority + map accuracy) decides.
        """
        manager = self.db_manager
        if manager is None:
            return None
        ids = list(getattr(manager, "active_source_ids", None) or [])
        if not ids:
            return None
        if getattr(self, "_fineness_weight", 0.0) > 0 and hasattr(
            manager, "effective_priority_steps"
        ):
            steps = manager.effective_priority_steps()
            return min(ids, key=lambda sid: (steps.get(sid, 0.0), sid))

        def rank(sid):
            src = manager.get_source_config(sid)
            return (getattr(src, "priority", 0) if src is not None else 0, sid)

        return min(ids, key=rank)

    def _main_before_patchify(self, rot, standard, expanded, source_id):
        """``(main RotationResult | None, own, patchify_only)`` when the main layer goes first.

        None keeps the old order (merged candidates of the chosen layer). ``own`` are
        the chosen layer's standard candidates, ``patchify_only`` the ones patchify
        added. ``main RotationResult`` is None when the chosen layer is the main one.
        """
        if self._patchify_main_margin <= 0 or self.db_manager is None or source_id is None:
            return None
        from src.database.source_priority import main_before_patchify

        std_ids = {int(frame) for frame, _ in standard}
        own = [c for c in expanded if int(c[0]) in std_ids]
        patch_only = [c for c in expanded if int(c[0]) not in std_ids]
        if not patch_only:
            return None
        main_sid = self._main_source_id()
        if main_sid is None:
            return None
        if main_sid == source_id:
            main_rot, main_candidates = None, standard
        else:
            main_rot = (getattr(rot, "per_source", None) or {}).get(main_sid)
            if main_rot is None or not main_rot.candidates:
                return None
            main_candidates = main_rot.candidates
        if not main_before_patchify(expanded, std_ids, main_candidates, self._patchify_main_margin):
            return None
        return main_rot, own, patch_only

    def _verify_main_layer(self, main_rot, query_frame, static_mask, cache):
        """Verify the main layer's candidates at its own (angle, scale); switch on success."""
        sid = main_rot.source_id
        database = self.db_manager.get_database(sid)
        if database is None:
            return None
        rotated, rot_mask, crop, feats = self._prepare_and_extract(
            query_frame,
            static_mask,
            main_rot.angle,
            main_rot.best_scale,
            cache,
            prepared=(main_rot.frame, main_rot.crop_info),
        )
        candidates = list(main_rot.candidates)
        support_check = getattr(database, "is_frame_georef_supported", None)
        if support_check is not None:
            candidates = [c for c in candidates if support_check(int(c[0]))]
        ver = self._geometric_verifier.verify(feats, candidates, database, tag="main_priority")
        if ver is None:
            return None
        self._active_source_id = sid
        self.database = database
        if self.calib_manager is not None:
            self.calibration = self.calib_manager.get(sid)
        logger.debug(f"Main layer '{sid}' verified before patchify candidates")
        return (
            ver,
            main_rot.angle,
            main_rot.best_scale,
            rotated,
            rot_mask,
            crop,
            feats,
            candidates,
            main_rot.score,
            sid,
        )

    def _gsd_of(self, source_id):
        """Ground pixel size of a layer (multi-layer: by id; single database: None)."""
        from src.database.source_priority import layer_gsd

        if self.db_manager is not None and source_id is not None:
            if hasattr(self.db_manager, "layer_gsd"):
                return self.db_manager.layer_gsd(source_id)
            database = self.db_manager.get_database(source_id)
        else:
            database = self.database
        if database is None:
            return None
        cached = self._layer_gsd_cache.get(id(database))
        if cached is None or cached[0] is not database:
            cached = (database, layer_gsd(database))
            self._layer_gsd_cache[id(database)] = cached
        return cached[1]

    def _estimated_query_gsd(self):
        """Query ground pixel size: scale prior x active layer's pixel size, or None.

        The scale prior is query / layer (sqrt of the last homography's scale, EMA).
        """
        prior = self._scale_manager.prior
        if prior is None:
            return None
        sid = self._active_source_id if self.db_manager is not None else None
        gsd = self._gsd_of(sid)
        return float(prior) * float(gsd) if gsd else None

    # ── localization.altitude_hint ──────────────────────────────────────────────

    def _layer_gsds(self, source_ids=None) -> dict:
        """Ground pixel size of the given (default: active) layers; one database: {None: it}."""
        if self.db_manager is None:
            return {None: self._gsd_of(None)}
        if source_ids is None:
            source_ids = list(getattr(self.db_manager, "active_source_ids", None) or [])
        return {sid: self._gsd_of(sid) for sid in source_ids}

    def _hint_query_gsd(self, norm_width, visual_gsd):
        """(query ground pixel size, source); telemetry needs metric (geo) layers."""
        hint = getattr(self, "_altitude_hint", None)
        if hint is None:
            return None, "off"
        relative = bool(getattr(self, "relative_mode", False))
        telemetry = None if relative else getattr(self, "_telemetry_altitude_m", None)
        gsd, source = hint.query_gsd(telemetry, norm_width, visual_gsd)
        logged = getattr(self, "_altitude_hint_logged", None)
        if gsd is not None and logged is not None and source not in logged:
            logged.add(source)
            logger.info(f"Altitude hint active ({source}): query ground pixel {gsd:.3f}")
        return gsd, source

    def _apply_altitude_hint(self, norm_width) -> None:
        """The nearest layers' expected scales go to the ScaleManager before retrieval."""
        hint = getattr(self, "_altitude_hint", None)
        if hint is None or not hint.enabled:
            return
        self._altitude_norm_width = norm_width
        gsd, source = self._hint_query_gsd(norm_width, getattr(self, "_frame_query_gsd", None))
        scales = hint.layer_scales(gsd, self._layer_gsds())
        self._scale_manager.set_altitude_scales([ratio for _, ratio in scales])
        if source == "telemetry":
            # localization.patchify_adaptive gates on the same estimate.
            self._frame_query_gsd = gsd
        if scales:
            listed = ", ".join(f"{sid}: x{ratio:.2f}" for sid, ratio in scales)
            logger.debug(f"Altitude hint ({source}): {listed}")

    def _altitude_layer_scales(self, norm_width) -> list:
        """Layer search: (source id, expected ratio) of the layers nearest the altitude."""
        hint = getattr(self, "_altitude_hint", None)
        if hint is None or not hint.enabled or self.db_manager is None:
            return []
        self._altitude_norm_width = norm_width
        search = self._layer_search
        active = search.handoff.active
        belief = search.beliefs.get(active) if active is not None else None
        active_gsd = self._gsd_of(active) if active is not None else None
        visual = None
        if belief is not None and active_gsd:
            visual = float(np.exp(belief.log_ratio)) * float(active_gsd)
        gsd, _ = self._hint_query_gsd(norm_width, visual)
        ids = getattr(self.db_manager, "all_source_ids", None)
        if ids is None:
            ids = getattr(self.db_manager, "active_source_ids", None)
        return hint.layer_scales(gsd, self._layer_gsds(list(ids or [])))

    def _learn_altitude_factor(self) -> None:
        """A fix made while telemetry was present refines the hint's camera footprint factor."""
        hint = getattr(self, "_altitude_hint", None)
        altitude = getattr(self, "_telemetry_altitude_m", None)
        if hint is None or not hint.enabled or altitude is None:
            return
        if getattr(self, "relative_mode", False):
            return
        width = getattr(self, "_altitude_norm_width", None)
        hint.learn(altitude, width, self._estimated_query_gsd())

    def _patchify_allowed(self, source_id) -> bool:
        """localization.patchify_adaptive: may this layer's patchify search run this frame?"""
        if not getattr(self, "_patchify_adaptive", False):
            return True
        from src.database.source_priority import grid_factors, patchify_layers

        if self.db_manager is not None:
            ids = list(getattr(self.db_manager, "active_source_ids", None) or [])
        else:
            ids = [None]
        layers = patchify_layers(
            getattr(self, "_frame_query_gsd", None),
            {sid: self._gsd_of(sid) for sid in ids},
            grid_factors(
                get_cfg(self.config, "localization.patchify_grids", [[1, 1], [2, 2], [3, 3]])
            ),
            self._patchify_gap_ratio,
        )
        return layers is None or source_id in layers

    def _layer_priorities(self) -> tuple[dict, dict]:
        """``(priority, patchify priority)`` per active layer (project.json numbers)."""
        from src.database.source_priority import effective_patchify_priority

        manager = self.db_manager
        standard: dict = {}
        patch: dict = {}
        for sid in list(getattr(manager, "active_source_ids", None) or []):
            src = manager.get_source_config(sid)
            standard[sid] = int(getattr(src, "priority", 0) or 0) if src is not None else 0
            patch[sid] = effective_patchify_priority(src) if src is not None else standard[sid]
        return standard, patch

    def _joint_candidates(self, per_source: dict, top_k: int) -> list:
        """Every layer's standard and patchify-only candidates (georeferenced frames only)."""
        from src.database.source_priority import JointCandidate

        entries = []
        desc_cache: dict = {}
        for sid, layer in per_source.items():
            database = self.db_manager.get_database(sid)
            if database is None:
                continue
            usable = getattr(database, "is_frame_georef_supported", None)
            standard_ids = {int(frame) for frame, _ in layer.candidates}
            for frame, score in layer.candidates:
                if usable is None or usable(int(frame)):
                    entries.append(JointCandidate(sid, int(frame), float(score), False))
            if not self._patchify_allowed(sid):
                continue
            for frame, score in self._candidate_retriever.patch_candidates(
                layer.frame, top_k, source_id=sid, desc_cache=desc_cache
            ):
                if int(frame) in standard_ids or (usable is not None and not usable(int(frame))):
                    continue
                entries.append(JointCandidate(sid, int(frame), float(score), True))
        return entries

    def _verify_joint(self, rot, query_frame, static_mask, cache, top_k):
        """Verify the joint candidate list in order; the first verified layer becomes active.

        Same return tuple as ``_verify_main_layer``, or None when nothing verified.
        Consecutive candidates of one layer go to the verifier together (its
        prefilter then works as before); each layer at its own (angle, scale).
        """
        from src.database.source_priority import joint_order, verification_runs

        per_source = rot.per_source or {}
        priorities, patch_priorities = self._layer_priorities()
        ordered = joint_order(
            self._joint_candidates(per_source, top_k),
            priorities,
            patch_priorities,
            self._joint_rank_step,
            self._joint_patch_penalty,
            self._joint_top_k,
        )
        for sid, run in verification_runs(ordered):
            layer = per_source.get(sid)
            database = self.db_manager.get_database(sid)
            if layer is None or database is None:
                continue
            rotated, rot_mask, crop, feats = self._prepare_and_extract(
                query_frame,
                static_mask,
                layer.angle,
                layer.best_scale,
                cache,
                prepared=(layer.frame, layer.crop_info),
            )
            candidates = [(c.frame_id, c.score) for c in run]
            tag = "joint_patchify" if all(c.patch for c in run) else "joint"
            ver = self._geometric_verifier.verify(feats, candidates, database, tag=tag)
            if ver is None:
                continue
            self._active_source_id = sid
            self.database = database
            if self.calib_manager is not None:
                self.calibration = self.calib_manager.get(sid)
            logger.debug(f"Joint candidate list: layer '{sid}' verified frame {ver.candidate_id}")
            return (
                ver,
                layer.angle,
                layer.best_scale,
                rotated,
                rot_mask,
                crop,
                feats,
                candidates,
                layer.score,
                sid,
            )
        return None

    def _build_patchify_retrieval(self, database):
        """Patch-descriptor index of one database, or None when it has none."""
        patch_desc = getattr(database, "patch_descriptors", None)
        if patch_desc is None or len(patch_desc) == 0:
            logger.info("Patchify enabled in config but database has no patch_descriptors. ")
            return None
        try:
            from src.localization.patchify import PatchifyRetrieval

            patchify_grids = get_cfg(
                self.config, "localization.patchify_grids", [[1, 1], [2, 2], [3, 3]]
            )
            patchify_batch = get_cfg(self.config, "localization.patchify_batch_size", 1)
            desc_dim = int(patch_desc.shape[-1])
            retrieval = PatchifyRetrieval(
                self.feature_extractor,
                descriptor_dim=desc_dim,
                grids=patchify_grids,
                batch_size=patchify_batch,
                unique_coverage=bool(
                    get_cfg(self.config, "localization.patchify_unique_coverage", False)
                ),
            )
            retrieval.build_index(patch_desc, list(range(database.get_num_frames())))
            logger.info(
                f"Patchify retrieval initialized: "
                f"{retrieval.num_patches} patches/frame, "
                f"dim={desc_dim}, grids={patchify_grids}"
            )
            return retrieval
        except Exception as e:
            logger.warning(
                f"Patchify retrieval init failed — falling back to standard retrieval: {e}"
            )
            return None

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
        self._of_steps_since_kf = 0
        self._consecutive_failures = 0
        self._last_best_angle = None
        self._last_state = None
        self._prev_fix_inliers = 0
        self._scale_manager.reset()
        self._layer_search.reset()
        self._layer_clock = None
        self._debug_depth_counter = 0
        self._tp_counter = 0
        self._tp_tries = 0
        self._tp_hits = 0
        self._spatial_prior.reset()
        self._sp_now = None
        self._sp_frame_done = False
        if self._smoother is not None:
            self._smoother.reset()

    def _maybe_set_depth_hint(self, frame: np.ndarray) -> None:
        """Soft depth-based reorder of the scale pyramid (every N keyframes; hint only)."""
        if not self._use_depth_hint or self._db_depth_scale is None:
            return
        depth_backend = get_cfg(self.config, "models.depth_estimator.backend", "depth_anything_v2")
        if depth_backend == "none":
            return
        self._depth_hint_counter += 1
        if (self._depth_hint_counter - 1) % max(1, self._depth_hint_every_n) != 0:
            return
        try:
            if self._depth_estimator is None:
                from src.depth.depth_estimator import DepthEstimator

                device = getattr(self.model_manager, "device", "cuda")
                self._depth_estimator = DepthEstimator.build(backend=depth_backend, device=device)
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
        query_frame,
        static_mask=None,
        dt=1.0,
        yaw_hint_deg=None,
        collector=None,
        timestamp=None,
        altitude_m=None,
    ) -> dict:
        # Height above ground / take-off from telemetry (localization.altitude_hint).
        self._telemetry_altitude_m = altitude_m
        self._frame_timestamp = None if timestamp is None else float(timestamp)
        enabled = get_cfg(self.config, "localization.layer_search.enabled", False)
        if enabled and self.db_manager is not None:
            return self._localize_layers(
                query_frame, static_mask, dt, yaw_hint_deg, collector, timestamp
            )
        previous = (self.database, self.calibration, self._active_source_id)
        sp = self._spatial_prior
        self._sp_now = sp.tick(timestamp, dt) if sp.enabled else None
        self._sp_frame_done = False
        had_prior = self._scale_manager.prior is not None
        # Feature / patch-descriptor caches of this frame, shared with the retry below.
        caches: dict = {}
        result = self._localize_frame_impl(
            query_frame, static_mask, dt, yaw_hint_deg, collector, _caches=caches
        )
        if not result.get("success"):
            self.database, self.calibration, self._active_source_id = previous
        if had_prior and (
            result.get("fallback_mode") == "retrieval_only"
            or str(result.get("error", "")).startswith("Not enough valid inliers")
        ):
            self._scale_manager.invalidate()
            self._last_best_angle = None
            result = self._localize_frame_impl(
                query_frame, static_mask, dt, yaw_hint_deg, collector, _caches=caches
            )
            if not result.get("success"):
                self.database, self.calibration, self._active_source_id = previous
        if result.get("fallback_mode") == "retrieval_only":
            self._consecutive_failures += 1
            self._scale_manager.invalidate()
        elif sp.enabled and result.get("success") and "lat" in result and "lon" in result:
            sp.record_fix(
                self._sp_now,
                float(result["lat"]),
                float(result["lon"]),
                angle=self._last_best_angle,
                scale=self._scale_manager.prior,
            )
        # localization.temporal_prior_kp_min_inliers: the next frame's prior check uses
        # the keypoint cap only after a strong fix.
        self._prev_fix_inliers = (
            int(result.get("inliers") or 0)
            if result.get("success") and not result.get("fallback_mode")
            else 0
        )
        if result.get("success") and not result.get("fallback_mode"):
            self._learn_altitude_factor()
        return self._coordinate_result(result)

    def _tp_keypoint_budget(self) -> int:
        """Query keypoints for this frame's temporal / spatial prior check (0 = all).

        localization.temporal_prior_max_keypoints, applied only when the previous
        frame's fix had at least localization.temporal_prior_kp_min_inliers inliers.
        """
        cap = int(getattr(self, "_tp_max_kp", 0) or 0)
        if cap <= 0:
            return 0
        need = int(getattr(self, "_tp_kp_min_inliers", 0) or 0)
        if need > 0 and int(getattr(self, "_prev_fix_inliers", 0) or 0) < need:
            return 0
        return cap

    def rebase_optical_flow_anchor(self, flow_affine) -> bool:
        """Move the OF anchor to a later frame (tracking.of_chain_on_keyframe_fail).

        ``flow_affine`` is the similarity (original px) from the current anchor
        frame to the new anchor frame, as measured by KeyframeFlow. The new
        frame's homography to the reference is H_new = H · Sc · S⁻¹ · Sc⁻¹ (Sc =
        resolution normalisation), i.e. a point of the new frame is mapped back
        into the old anchor frame and then through the old H. Only for states in
        the unrotated frame (continuous rotation mode stores angle 0); returns
        False otherwise and the caller keeps the legacy behaviour.
        """
        state = self._last_state
        if state is None or state.get("H") is None or flow_affine is None:
            return False
        if int(round(float(state.get("global_angle", 0) or 0))) % 360 != 0:
            return False
        S = np.asarray(flow_affine, dtype=np.float64)
        if S.shape != (2, 3) or not np.all(np.isfinite(S)):
            return False
        scale = float(state.get("scale", self._last_scale) or 1.0)
        S3 = np.vstack([S, [0.0, 0.0, 1.0]])
        try:
            S_inv = np.linalg.inv(S3)
        except np.linalg.LinAlgError:
            return False
        Sc = np.diag([scale, scale, 1.0])
        Sc_inv = np.diag([1.0 / scale, 1.0 / scale, 1.0])
        H_new = np.asarray(state["H"], dtype=np.float64) @ Sc @ S_inv @ Sc_inv
        if not np.all(np.isfinite(H_new)):
            return False
        new_state = dict(state)
        new_state["H"] = H_new
        new_state["of_chain_links"] = int(state.get("of_chain_links", 0)) + 1
        self._last_state = new_state
        self._last_of_raw = None
        self._of_steps_since_kf = 0
        return True

    def _hint_allows_temporal_prior(self, yaw_hint_deg: float | None) -> bool:
        """True when the temporal prior may run under this yaw hint: no hint,
        or (flag on) a hint within the tolerance of the last fix's angle."""
        if yaw_hint_deg is None:
            return True
        if self._tp_hint_tol is None:
            return False
        if self._last_best_angle is None:
            # No angle to disagree with (after a reset / failures). The prior
            # checks its own preconditions (last state, dead-reckoned
            # position); blocking it here also blocked the spatial prior and
            # kept failing runs failing (wayback 2019 with heading: 0.47).
            return True
        hint = norm_deg(yaw_hint_deg)
        if not self._rot_continuous:
            hint = float((int(round(hint / 90.0)) * 90) % 360)
        diff = abs(hint - norm_deg(float(self._last_best_angle))) % 360.0
        return min(diff, 360.0 - diff) <= self._tp_hint_tol

    def _filter_dt(self, dt: float) -> float:
        """KF prediction interval: time since the filter's last update when
        tracking.kalman_dt_from_timestamps is on and both timestamps are known;
        otherwise the caller's dt (legacy)."""
        if not self._kf_dt_from_ts or self._frame_timestamp is None:
            return dt
        last = self.trajectory_filter.last_update_t
        if last is None:
            return dt
        elapsed = self._frame_timestamp - last
        return elapsed if elapsed > 0.0 else dt

    def _coordinate_result(self, result):
        if self.relative_mode:
            result["coordinate_kind"] = "local_planar"
            result["coordinate_units"] = "arbitrary"
            if "lat" in result and "lon" in result:
                # lat/lon remain internal vertical/horizontal transport for the
                # existing Qt signals. Network and exports emit x/y only.
                result["x"], result["y"] = result["lon"], result["lat"]
        return result

    def _localize_layers(self, query_frame, static_mask, dt, yaw_hint, collector, timestamp):
        from copy import deepcopy

        if self.calib_manager is None:
            return {
                "success": False,
                "status": "lost",
                "error": "Layer calibration manager missing",
            }
        now = (
            (self._layer_clock or 0.0) + max(0.0, float(dt))
            if timestamp is None
            else float(timestamp)
        )
        if not np.isfinite(now) or (self._layer_clock is not None and now <= self._layer_clock):
            return {"success": False, "status": "stale", "error": "Non-increasing frame timestamp"}
        self._layer_clock = now
        frame, _ = self.normalizer.normalize(query_frame)
        mask = self.normalizer.normalize_mask(static_mask) if static_mask is not None else None
        altitude_layers = self._altitude_layer_scales(frame.shape[1])
        observations = self._layer_search.search(
            self,
            frame,
            mask,
            now,
            yaw_hint,
            **({"altitude_layers": altitude_layers} if altitude_layers else {}),
        )
        chosen = self._layer_search.handoff.choose(observations, now)
        diagnostics = dict(self._layer_search.last_diagnostics)
        if chosen is None:
            self._consecutive_failures += 1
            if (
                self._layer_search.handoff.state == "LOST"
                or self._layer_search.handoff.reason == "ambiguous_geometry"
            ):
                self._last_state = None
            return {
                "success": False,
                "status": "ambiguous"
                if self._layer_search.handoff.reason == "ambiguous_geometry"
                else "lost",
                "error": self._layer_search.handoff.reason,
                "layer_state": self._layer_search.handoff.state,
                "search": diagnostics,
            }
        previous = (self.database, self.calibration, self._active_source_id, self._scale_manager)
        switching = chosen.source_id != self._active_source_id
        filters = (self.trajectory_filter, self.outlier_detector, self._smoother)
        previous_state = (self._last_state, self._last_best_angle, self._last_of_raw)
        self.trajectory_filter = deepcopy(self.trajectory_filter)
        self.outlier_detector = deepcopy(self.outlier_detector)
        self._scale_manager = deepcopy(self._scale_manager)
        if self._smoother is not None:
            self._smoother = deepcopy(self._smoother)
        self.database = self.db_manager.get_database(chosen.source_id)
        self.calibration = self.calib_manager.get(chosen.source_id)
        self._active_source_id = chosen.source_id
        if switching:
            # Metric projections can differ between layers. Start a fresh filter
            # only after same-frame geodesic handoff checks; never mix raw units.
            self.trajectory_filter.reset()
            self.outlier_detector.reset()
            if self._smoother is not None:
                self._smoother.reset()
            self._scale_manager = ScaleManager(self.config)
        accepted = False
        try:
            result = self._localize_frame_impl(
                query_frame, static_mask, dt, yaw_hint, collector, _verified=chosen.prepared
            )
            accepted = bool(result.get("success"))
            if accepted:
                self._layer_search.commit(chosen, now)
                self._learn_altitude_factor()
                result["status"] = "confirmed"
                result["source_changed"] = switching
                result["coordinate_kind"] = "ground_observation"
            result["layer_state"] = self._layer_search.handoff.state
            result["search"] = diagnostics
            return result
        finally:
            if not accepted:
                self.database, self.calibration, self._active_source_id, self._scale_manager = (
                    previous
                )
                self.trajectory_filter, self.outlier_detector, self._smoother = filters
                self._last_state, self._last_best_angle, self._last_of_raw = previous_state

    def _localize_frame_impl(
        self,
        query_frame: np.ndarray,
        static_mask: np.ndarray = None,
        dt: float = 1.0,
        yaw_hint_deg: float | None = None,
        collector=None,
        _verified=None,
        _caches: dict | None = None,
    ) -> dict:
        # Fix #1: If too many consecutive failures occurred — return out_of_coverage
        if _verified is None and self._consecutive_failures >= self._max_failures:
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
        if _verified is None:
            self._maybe_set_depth_hint(query_frame)
        # Debug: depth map for window (independent of localization success).
        self._maybe_collect_depth(query_frame, collector)

        if not self.enable_auto_rotation:
            angles_to_try = [0]
        elif self._rot_continuous:
            angles_to_try = list(self._rot_scan)
        else:
            angles_to_try = [0, 90, 180, 270]

        top_k = self.retrieval_top_k

        # §A1: cache (angle, scale) -> (frame, mask, crop, features) for ONE frame.
        # If temporal hypothesis failed, and full path chose the same
        # angle and scale — ALIKED is not recomputed (217-339 ms on GTX 1650).
        # ``_caches`` (localize_frame) carries it, and the query's patchify
        # descriptors, into the in-frame retry, which used to recompute both for
        # the view it had already processed.
        caches = {} if _caches is None else _caches
        _feat_cache: dict = caches.setdefault("features", {})
        _patch_cache: dict = caches.setdefault("patch", {})

        # ── §A1: attempt to localize WITHOUT global descriptor ──────────
        # yaw_hint_deg disables this path: external heading is new information
        # about orientation, it must be processed by full rotation path.
        _tp = _verified
        # True only for a temporal-prior hit (not layer search's _verified): the
        # rotation refine then compares inliers under the same keypoint budget.
        tp_hit = False
        if _tp is None and self._temporal_prior and self._hint_allows_temporal_prior(yaw_hint_deg):
            self._tp_counter += 1
            audit = self._tp_audit_every
            if audit <= 0 or (self._tp_counter % audit) != 0:
                self._tp_tries += 1
                _tp = self._try_temporal_prior(query_frame, static_mask, _feat_cache)
                if _tp is not None:
                    self._tp_hits += 1
                    tp_hit = True
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
                collector.global_angle = self._angle_out(best_global_angle)
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
            # yaw_hint_deg — angle to rotate the frame by to match the DB
            # orientation, in the np.rot90 / cv2.getRotationMatrix2D sense
            # (positive = counter-clockwise on screen); src/telemetry converts a
            # compass heading into it. "quarter" mode quantizes it to k*90;
            # "continuous" uses it as is. A false hint self-heals: if the
            # retrieval score at the prior angle is below rotation_rescan_min_score,
            # RotationSelector falls back to the full scan.
            # Query scale vs the layer of the last fix, before retrieval switches layers
            # (localization.patchify_adaptive).
            self._frame_query_gsd = self._estimated_query_gsd()
            self._apply_altitude_hint(width)
            prior_angle = self._last_best_angle
            use_prior = self.enable_auto_rotation and self._consecutive_failures == 0
            if yaw_hint_deg is not None and self.enable_auto_rotation:
                if self._rot_continuous:
                    prior_angle = norm_deg(yaw_hint_deg)
                else:
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
                collector.global_angle = self._angle_out(best_global_angle)
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
            standard_candidates = list(best_global_candidates)
            # Joint candidate list: every layer's patchify candidates are searched there.
            joint_mode = (
                self._joint_ranking
                and self.db_manager is not None
                and bool(getattr(rot, "per_source", None))
            )
            # localization.patchify_lazy: the patch search waits until the standard
            # candidates of this view have failed (see the lazy branch below).
            lazy_patchify = (
                getattr(self, "_patchify_lazy", False)
                and not joint_mode
                and self._candidate_retriever.patchify_for(best_source_id_per_angle) is not None
            )
            if (
                not joint_mode
                and not lazy_patchify
                and self._patchify_allowed(best_source_id_per_angle)
            ):
                best_global_candidates = self._candidate_retriever.expand(
                    best_rotated_frame,
                    best_global_candidates,
                    top_k,
                    source_id=best_source_id_per_angle,
                    desc_cache=_patch_cache,
                )

            # Map trust participates in candidate selection, not only in final
            # result assembly.  Otherwise an unsupported self-match can win the
            # verifier and hide a slightly weaker supported candidate.
            support_check = getattr(self.database, "is_frame_georef_supported", None)
            if support_check is not None:
                best_global_candidates = [
                    candidate
                    for candidate in best_global_candidates
                    if support_check(int(candidate[0]))
                ]

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

            main_first = (
                None
                if joint_mode or lazy_patchify
                else self._main_before_patchify(
                    rot, standard_candidates, best_global_candidates, best_source_id_per_angle
                )
            )
            if joint_mode:
                ver = None
                on_layer = self._verify_joint(rot, query_frame, static_mask, _feat_cache, top_k)
                if on_layer is not None:
                    (
                        ver,
                        best_global_angle,
                        best_scale,
                        best_rotated_frame,
                        best_rotated_mask,
                        _crop_info,
                        best_query_features,
                        best_global_candidates,
                        best_global_score,
                        best_source_id_per_angle,
                    ) = on_layer
            elif lazy_patchify:
                ver = self._geometric_verifier.verify(
                    best_query_features, best_global_candidates, self.database, tag="retrieval"
                )
                if ver is None and self._patchify_allowed(best_source_id_per_angle):
                    expanded = self._candidate_retriever.expand(
                        best_rotated_frame,
                        standard_candidates,
                        top_k,
                        source_id=best_source_id_per_angle,
                        desc_cache=_patch_cache,
                    )
                    if support_check is not None:
                        expanded = [c for c in expanded if support_check(int(c[0]))]
                    std_ids = {int(frame) for frame, _ in standard_candidates}
                    patch_only = [c for c in expanded if int(c[0]) not in std_ids]
                    if patch_only:
                        # Same order as the eager path from here on: the main layer
                        # (patchify_main_rank_margin), then the patchify-only candidates.
                        best_global_candidates = expanded
                        lazy_first = self._main_before_patchify(
                            rot, standard_candidates, expanded, best_source_id_per_angle
                        )
                        main_rot = lazy_first[0] if lazy_first is not None else None
                        if main_rot is not None:
                            on_main = self._verify_main_layer(
                                main_rot, query_frame, static_mask, _feat_cache
                            )
                            if on_main is not None:
                                (
                                    ver,
                                    best_global_angle,
                                    best_scale,
                                    best_rotated_frame,
                                    best_rotated_mask,
                                    _crop_info,
                                    best_query_features,
                                    best_global_candidates,
                                    best_global_score,
                                    best_source_id_per_angle,
                                ) = on_main
                        if ver is None:
                            ver = self._geometric_verifier.verify(
                                best_query_features, patch_only, self.database, tag="patchify"
                            )
            elif main_first is None:
                ver = self._geometric_verifier.verify(
                    best_query_features, best_global_candidates, self.database, tag="retrieval"
                )
            else:
                # Patchify-only candidates come last: this layer's standard candidates,
                # then the main layer's (another layer: switch + its own scale), then them.
                main_rot, own, patch_only = main_first
                ver = self._geometric_verifier.verify(
                    best_query_features, own, self.database, tag="retrieval"
                )
                if ver is None and main_rot is not None:
                    on_main = self._verify_main_layer(
                        main_rot, query_frame, static_mask, _feat_cache
                    )
                    if on_main is not None:
                        (
                            ver,
                            best_global_angle,
                            best_scale,
                            best_rotated_frame,
                            best_rotated_mask,
                            _crop_info,
                            best_query_features,
                            best_global_candidates,
                            best_global_score,
                            best_source_id_per_angle,
                        ) = on_main
                if ver is None:
                    ver = self._geometric_verifier.verify(
                        best_query_features, patch_only, self.database, tag="patchify"
                    )

        # ── Continuous rotation: remove the residual the scan step left ──────
        # The scan lands within step/2 of the true heading; the verified
        # homography measures the rest. One re-extraction at the corrected angle
        # turns e.g. a 20° residual (~half the inliers, see query_rotation) into
        # ~0°. Kept only if it verifies the same frame with more inliers.
        if (
            ver is not None
            and self._rot_continuous
            and self.enable_auto_rotation
            and self._rot_refine_min > 0
        ):
            refined = self._refine_rotation(
                query_frame,
                static_mask,
                best_global_angle,
                best_scale,
                ver,
                _feat_cache,
                max_keypoints=self._tp_keypoint_budget() if tp_hit else 0,
            )
            if refined is not None:
                (
                    ver,
                    best_global_angle,
                    best_rotated_frame,
                    best_rotated_mask,
                    _crop_info,
                    best_query_features,
                ) = refined
                if collector is not None:
                    collector.global_angle = self._angle_out(best_global_angle)
                    collector.rotated_frame = best_rotated_frame
                    collector.query_features = best_query_features
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
        support_check = getattr(self.database, "is_frame_georef_supported", None)
        if support_check is not None and not support_check(best_candidate_id):
            self._log_failure(
                FAILURE_TYPES["No propagated calibration"],
                details=f"Frame {best_candidate_id} georeference is not visually supported",
            )
            return {
                "success": False,
                "error": (
                    f"Frame {best_candidate_id} has provisional or invalid georeference; "
                    "GPS output suppressed"
                ),
            }
        from src.geometry.frame_transform import get_frame_transform

        affine_ref = get_frame_transform(self.database, best_candidate_id)
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

        # Dimensions of rotated normalized image. Continuous mode keeps the
        # view the size of the frame and folds the rotation into H below, so
        # everything downstream works in ORIGINAL frame coordinates.
        if not self._rot_continuous and best_global_angle in (90, 270):
            rot_height, rot_width = width, height
        else:
            rot_height, rot_width = height, width
        # Rotation measured by this match (view -> reference), for the next prior.
        H_view_to_ref = best_H_query_to_ref

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
        if _crop_info is not None:
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

        # Continuous mode: view -> original frame. After this, M maps ORIGINAL
        # (resolution-normalised) frame pixels to the reference frame and the
        # rest of the pipeline treats the keyframe as unrotated (angle 0).
        downstream_angle = best_global_angle
        if self._rot_continuous:
            R_view = rotation_matrix(best_global_angle, width, height)
            M_query_to_ref = M_query_to_ref @ R_view
            if best_mkpts_q_inliers is not None and len(best_mkpts_q_inliers) > 0:
                best_mkpts_q_inliers = GeometryTransforms.apply_homography(
                    np.asarray(best_mkpts_q_inliers, dtype=np.float64), np.linalg.inv(R_view)
                )
            downstream_angle = 0

        # ── Step 5: State for Optical Flow (commit — AFTER outlier gate) ─────
        ref_h, ref_w = self.database.get_frame_size(best_candidate_id)
        pending_state = {
            "reference_size": (ref_w, ref_h),
            "H": M_query_to_ref,
            "affine": affine_ref,
            "candidate_id": best_candidate_id,
            "inliers": best_inliers,
            "global_angle": downstream_angle,
            "source_id": self._active_source_id,
            # Normalization scale of THIS SPECIFIC keyframe: OF operates in
            # frame system belonging to H (fresh self._last_scale on
            # subsequent frames may already differ).
            "scale": self._last_scale,
            # Frame time of this real keyframe fix (dead-reckoning age for OF).
            "fix_t": self._frame_timestamp,
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

        try:
            pts_metric = project_frame_points(affine_ref, pts_in_ref, ref_w, ref_h)
        except ValueError:
            return {"success": False, "error": "Center outside safe reference projection"}
        mx = float(pts_metric[0, 0])
        my = float(pts_metric[0, 1])
        metric_pt = np.array([mx, my], dtype=np.float64)

        # ── Step 7: Outlier filtering ─────────────────────────────────────
        # Strong geometry beats kinematic prior: fix supported by hundreds of
        # RANSAC inliers should not be discarded due to platform speed assumptions.
        # Position is still appended to history so detector window
        # corresponded to reality, not to a filtered version of it.
        _strong = self.relative_mode or (
            self._trust_strong and best_inliers >= self._trust_min_inliers
        )
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
        _kf_dt = self._filter_dt(dt)
        _maha_d2 = (
            self.trajectory_filter.mahalanobis_sq(metric_pt, dt=_kf_dt, noise_scale=1.0)
            if self._maha_gate_enabled
            else None
        )
        if not _strong and self.outlier_detector.is_outlier(
            metric_pt,
            dt,
            maha_d2=_maha_d2,
            pred_position=self.trajectory_filter.predict_position(_kf_dt),
        ):
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

        filtered_pt = (
            metric_pt
            if self.relative_mode
            else self.trajectory_filter.update(
                metric_pt,
                dt=_kf_dt,
                noise_scale=1.0 / max(confidence, 0.25),
                trusted_max_offset_m=(self._trusted_fix_max_filter_offset if _strong else None),
                timestamp=self._frame_timestamp,
            )
        )
        # RESEARCH 3.1: back-end smoother — fix window + OF-odometry;
        # KF correction by shift BEFORE writing to detector history and GPS/FOV,
        # so correction lands in THIS frame.
        if self._smoother is not None:
            if self.trajectory_filter.last_update_reanchored:
                # All old smoother nodes describe the superseded motion
                # regime; keeping them would immediately pull the fresh KF
                # anchor back toward the turn's stale constant-velocity path.
                self._smoother.reset()
            corr = self._smoother.add_fix(
                metric_pt,
                dt=dt,
                confidence=confidence,
                source_id=self._active_source_id,
                accepted=True,
                kf_xy=filtered_pt,
            )
            if corr is not None:
                proposed_pt = (
                    float(filtered_pt[0]) + float(corr[0]),
                    float(filtered_pt[1]) + float(corr[1]),
                )
                proposed_offset = float(np.linalg.norm(np.asarray(proposed_pt) - metric_pt))
                if _strong and proposed_offset > self._trusted_fix_max_filter_offset:
                    # Never let a delayed smoother correction violate the
                    # same contract enforced by the front-end KF.
                    self._smoother.reset()
                    logger.debug(
                        "Smoother correction discarded: trusted-fix offset "
                        f"would be {proposed_offset:.2f} m"
                    )
                else:
                    self.trajectory_filter.shift(float(corr[0]), float(corr[1]))
                    filtered_pt = proposed_pt
                    logger.debug(f"Smoother correction applied: ({corr[0]:+.2f}, {corr[1]:+.2f}) m")
        self.outlier_detector.add_position(filtered_pt, dt=dt)
        # New keyframe restarts LK, so chain of local OF comparisons
        # breaks: first OF after keyframe should measure shift FROM keyframe
        # (ref=None -> database = newly added position in window), not from OF-measurement
        # of previous cycle. Otherwise database of shift and dt diverge again.
        self._last_of_raw = None
        self._of_steps_since_kf = 0
        lat, lon = self.calibration.converter.metric_to_gps(
            float(filtered_pt[0]), float(filtered_pt[1])
        )
        raw_lat, raw_lon = self.calibration.converter.metric_to_gps(mx, my)
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
            reference_size=(ref_w, ref_h),
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
        if self._rot_continuous:
            # Angle at which this match would have had zero residual rotation:
            # in steady flight the next keyframe is then aligned to within the
            # heading change between two keyframes.
            self._last_best_angle = next_prior_deg(best_global_angle, H_view_to_ref)
        else:
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
            "raw_lat": raw_lat,
            "raw_lon": raw_lon,
            "raw_metric": [mx, my],
            "fix_source": "keyframe",
            "since_fix_s": 0.0,
            "scale_ratio": float(best_scale),
            "rotation_deg": self._angle_out(best_global_angle),
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
        timestamp: float | None = None,
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

        try:
            pts_metric = project_frame_points(
                self._last_state["affine"],
                pts_in_ref,
                *self._last_state.get("reference_size", (0, 0)),
            )
        except ValueError:
            return {"success": False, "error": "OF center outside safe reference projection"}
        if pts_metric is None or len(pts_metric) == 0:
            return {"success": False, "error": "OF affine failed"}

        mx, my = float(pts_metric[0, 0]), float(pts_metric[0, 1])
        metric_pt = np.array([mx, my], dtype=np.float64)

        if (
            self._of_min_quality > 0.0
            and flow_quality is not None
            and float(flow_quality) < self._of_min_quality
        ):
            logger.warning(
                f"OF rejected: flow_quality={float(flow_quality):.3f} < "
                f"{self._of_min_quality} (LK track lost)"
            )
            self._log_failure(FAILURE_TYPES["Outlier detected"])
            return {"success": False, "error": "OF rejected — flow quality too low"}

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
        self._of_steps_since_kf += 1
        _of_corr = 1.0 + self._of_noise_growth * (self._of_steps_since_kf - 1)
        # noise_scale=1.5 matches base multiplier of OF-branch update() at
        # of_conf=1.0 — OF measurement is inherently noisier than keyframe-fix.
        self._frame_timestamp = None if timestamp is None else float(timestamp)
        _kf_dt = self._filter_dt(dt)
        _maha_d2 = (
            self.trajectory_filter.mahalanobis_sq(
                metric_pt, dt=_kf_dt, noise_scale=self._of_noise_base * _of_corr
            )
            if self._maha_gate_enabled
            else None
        )
        _is_out = (
            not self.relative_mode
            and self._of_outlier_gate
            and not _strong_flow
            and self.outlier_detector.is_outlier(
                metric_pt,
                dt,
                ref_position=_of_ref,
                maha_d2=_maha_d2,
                pred_position=self.trajectory_filter.predict_position(_kf_dt),
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

        filtered_pt = (
            metric_pt
            if self.relative_mode
            else self.trajectory_filter.update(
                metric_pt,
                dt=_kf_dt,
                noise_scale=self._of_noise_base * _of_corr / max(of_conf, 0.25),
                timestamp=self._frame_timestamp,
            )
        )
        self.outlier_detector.add_position(filtered_pt, dt=dt, reset_consecutive=False)

        lat, lon = self.calibration.converter.metric_to_gps(
            float(filtered_pt[0]), float(filtered_pt[1])
        )
        self._sync_ground_scale(lat)
        # Raw OF position (before KF / smoother), for diagnostics and replay
        # scoring — same meaning as raw_lat/raw_lon of a keyframe result.
        raw_lat, raw_lon = self.calibration.converter.metric_to_gps(mx, my)

        of_inliers = int(self._last_state.get("inliers", 30) * 0.8)
        # Provenance for consumers: an OF position chained through failed
        # keyframes (tracking.of_chain_on_keyframe_fail) is dead reckoning, not
        # an observation of the map — flagged, with its age and halved confidence.
        links = int(self._last_state.get("of_chain_links", 0) or 0)
        fix_t = self._last_state.get("fix_t")
        since_fix = (
            round(self._frame_timestamp - float(fix_t), 2)
            if fix_t is not None and self._frame_timestamp is not None
            else None
        )
        if links > 0:
            of_conf *= 0.5

        return self._coordinate_result(
            {
                "success": True,
                "lat": lat,
                "lon": lon,
                "raw_lat": raw_lat,
                "raw_lon": raw_lon,
                "fix_source": "dead_reckoning" if links > 0 else "optical_flow",
                "dead_reckoning_links": links,
                "since_fix_s": since_fix,
                "confidence": round(of_conf, 3),
                "matched_frame": int(self._last_state.get("candidate_id", -1)),
                "inliers": of_inliers,
                "fov_polygon": None,
                "is_of": True,
            }
        )

    # ─────────────────────────────────────────────────────────────────────────

    # ── Continuous rotation helpers (localization.rotation_mode) ─────────────

    def _angle_out(self, angle) -> float | int:
        """Angle as reported/collected: int (k*90) in quarter mode, float otherwise."""
        return round(float(angle), 2) if self._rot_continuous else int(angle)

    def _refine_rotation(self, query_frame, static_mask, angle, scale, ver, cache, max_keypoints=0):
        """Re-verify ``ver``'s frame at the residual-corrected angle.

        Returns ``(ver, angle, frame, mask, crop_info, features)`` when the
        corrected view verifies the same reference frame with more inliers,
        else None. ``max_keypoints`` (0 = all) must be the budget ``ver`` was
        verified with, so the inlier counts are comparable.
        """
        target = next_prior_deg(angle, ver.H_query_to_ref)
        residual = angle_distance_deg(target, angle)
        if residual < self._rot_refine_min:
            return None
        frame, mask, crop, feats = self._prepare_and_extract(
            query_frame, static_mask, target, scale, cache
        )
        ver2 = self._geometric_verifier.verify(
            top_keypoints(feats, max_keypoints),
            [(int(ver.candidate_id), 0.0)],
            self.database,
            tag="refine",
        )
        if ver2 is None or ver2.inliers <= ver.inliers:
            logger.debug(
                f"Rotation refine {angle:.1f}° -> {target:.1f}° kept the original "
                f"({ver.inliers} vs {getattr(ver2, 'inliers', 0)} inliers)"
            )
            return None
        logger.debug(
            f"Rotation refine {angle:.1f}° -> {target:.1f}° (residual {residual:.1f}°): "
            f"inliers {ver.inliers} -> {ver2.inliers}"
        )
        return ver2, target, frame, mask, crop, feats

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
        if self._rot_continuous:
            return self._prepare_and_extract_view(
                query_frame, static_mask, angle, scale, cache, prepared
            )
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
        # ``image_size`` is part of the local-feature contract used both by
        # LightGlue normalisation and by the spatial-support safety gate.  Keep
        # compatibility with lightweight/custom extractors that only return
        # keypoints and descriptors: the Localizer owns the prepared image and
        # therefore knows this value exactly.
        feats.setdefault(
            "image_size", np.array([rotated.shape[0], rotated.shape[1]], dtype=np.int32)
        )
        cache[key] = (rotated, rot_mask, crop_info, feats)
        return cache[key]

    def _prepare_and_extract_view(
        self,
        query_frame: np.ndarray,
        static_mask: np.ndarray | None,
        angle: float,
        scale: float,
        cache: dict,
        prepared: tuple | None = None,
    ) -> tuple:
        """Continuous-mode twin of ``_prepare_and_extract``.

        Same-size rotated view (query_rotation.rotate_view) → GSD normalisation
        → ALIKED, with keypoints restricted to the part of the view that comes
        from the frame (and to the static/YOLO mask).
        """
        key = ("view", round(float(angle), 2), round(float(scale), 3))
        cached = cache.get(key)
        if cached is not None:
            return cached
        from src.localization.rotation_selector import RotationSelector

        with Telemetry.profile("view_prepare"):
            if prepared is not None and prepared[0] is not None:
                frame, crop_info = prepared[0], prepared[1]
            else:
                frame, crop_info, _ = RotationSelector._prepare_view(
                    query_frame, angle, scale, self._scale_manager, None, self._rot_erode
                )
            mask = view_mask(query_frame.shape[:2], angle, static_mask, self._rot_erode)
            if mask is not None and abs(float(scale) - 1.0) > 0.15:
                mask, _ = self._scale_manager.normalize(mask, float(scale))
        feats = self.feature_extractor.extract_local_features(frame, static_mask=mask)
        feats.setdefault("image_size", np.array([frame.shape[0], frame.shape[1]], dtype=np.int32))
        cache[key] = (frame, mask, crop_info, feats)
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
                if self._tp_frame_usable(c):
                    ids.append(c)
        return ids

    def _tp_frame_usable(self, c: int) -> bool:
        """A frame can end a temporal / spatial prior check: georeferenced + calibrated."""
        try:
            support_check = getattr(self.database, "is_frame_georef_supported", None)
            if support_check is not None and not support_check(c):
                return False
            return self.database.get_frame_affine(c) is not None
        except Exception as e:  # noqa: BLE001 — БД може не мати кадру
            logger.debug(f"Temporal prior: frame {c} rejected ({e})")
            return False

    def _try_temporal_prior(
        self, query_frame: np.ndarray, static_mask: np.ndarray | None, cache: dict
    ) -> tuple | None:
        """Index-neighbour temporal prior, then (flag) the spatial prior."""
        self._tp_verified = []
        hit = self._try_index_prior(query_frame, static_mask, cache)
        if hit is None and self._spatial_prior.enabled:
            hit = self._try_spatial_prior(query_frame, static_mask, cache)
        return hit

    def _try_spatial_prior(
        self, query_frame: np.ndarray, static_mask: np.ndarray | None, cache: dict
    ) -> tuple | None:
        """Second temporal-prior stage: map frames near the dead-reckoned position.

        Runs at most once per ``localize_frame`` call (the in-frame retry would
        repeat it) and only within ``spatial_prior_max_age_s`` of the last fix.
        Angle and scale fall back to the last confirmed ones (the retry clears
        both priors). Candidates: the ``spatial_prior_nearest_k`` usable frames
        of the active layer nearest to the prediction, then the index
        neighbours; the best ``spatial_prior_verify`` by MNN go to LightGlue
        without the ``temporal_prior_min_mnn`` gate (on other-satellite imagery
        the probe scores the right frame below it while LightGlue accepts it).
        Same return tuple as ``_try_index_prior``; verification tag
        ``spatial_prior``.
        """
        if self._sp_frame_done or self._sp_now is None:
            return None
        self._sp_frame_done = True
        sp = self._spatial_prior
        pred = sp.predict(self._sp_now)
        if pred is None:
            return None
        angle = self._last_best_angle if self._last_best_angle is not None else sp.angle
        if angle is None:
            return None
        scale = self._scale_manager.prior
        if scale is None:
            scale = sp.scale
        scale = 1.0 if scale is None else float(scale)

        index_ids = self._tp_neighbour_ids()
        ids = sp.nearest_ids(
            self.database, pred[0], pred[1], exclude=index_ids, usable=self._tp_frame_usable
        )
        # Frames the index stage already ran LightGlue on in this frame are not
        # verified twice.
        ids = [c for c in ids + index_ids if c not in self._tp_verified]
        if not ids:
            return None
        self._sp_tries += 1

        rotated, rot_mask, crop_info, feats = self._prepare_and_extract(
            query_frame, static_mask, angle, scale, cache
        )
        tp_feats = top_keypoints(feats, self._tp_keypoint_budget())
        cands = [(int(i), 0.0) for i in ids]
        ref_cache: dict = {}
        scored = self._geometric_verifier.mnn_counts(tp_feats, cands, self.database, ref_cache)
        if not scored:
            return None
        # Stable sort: equal probe counts keep the nearest-first order.
        scored.sort(key=lambda t: -t[0])
        probe = [(cid, float(m)) for m, cid, _ in scored[: sp.verify]]
        ver = self._geometric_verifier.verify(
            tp_feats, probe, self.database, ref_cache=ref_cache, tag="spatial_prior"
        )
        if ver is None or ver.inliers < self._tp_accept:
            got = ver.inliers if ver is not None else 0
            logger.debug(
                f"Spatial prior: rejected ({got} inliers < {self._tp_accept}; "
                f"{len(ids)} candidates, age {sp.age(self._sp_now):.1f} s)"
            )
            return None
        self._sp_hits += 1
        if self._sp_tries % 25 == 0 or self._sp_hits == 1:
            logger.info(f"[spatial-prior] tries={self._sp_tries} hits={self._sp_hits}")
        logger.debug(
            f"Spatial prior HIT: frame={ver.candidate_id}, inliers={ver.inliers}, "
            f"angle={angle:.1f} deg, age {sp.age(self._sp_now):.1f} s"
        )
        out_angle = float(angle) if self._rot_continuous else int(angle)
        return (ver, out_angle, float(scale), rotated, rot_mask, crop_info, feats, probe)

    def _try_index_prior(
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
        # Probe and verification on the strongest keypoints only; `feats` (the
        # full set, also in `cache` for the retrieval path) is what is returned.
        tp_feats = top_keypoints(feats, self._tp_keypoint_budget())

        cands = [(int(i), 0.0) for i in ids]
        ref_cache: dict = {}
        scored = self._geometric_verifier.mnn_counts(tp_feats, cands, self.database, ref_cache)
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
        self._tp_verified = [cid for cid, _ in probe]
        ver = self._geometric_verifier.verify(
            tp_feats, probe, self.database, ref_cache=ref_cache, tag="temporal_prior"
        )
        if ver is None or ver.inliers < self._tp_accept:
            got = ver.inliers if ver is not None else 0
            logger.debug(
                f"Temporal prior: rejected ({got} inliers < {self._tp_accept}) — full path"
            )
            return None

        out_angle = float(angle) if self._rot_continuous else int(angle)
        return (ver, out_angle, float(scale), rotated, rot_mask, crop_info, feats, probe)

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
                    max_iters=self.ransac_max_iters,
                    confidence=self.ransac_confidence,
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
