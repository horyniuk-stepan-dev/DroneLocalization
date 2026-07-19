"""
Графова пропагація калібрування координат.

Замість лінійного ланцюжка гомографій, будується граф кадрів із:
  - Часовими ребрами (sequential: frame i ↔ frame i+1)
  - Просторовими ребрами (loop closure: DINOv2 retrieval → LightGlue matching)
  - GPS-якорями як жорсткими вузлами

Оптимізація: Levenberg-Marquardt через scipy.optimize.least_squares
з SO(2)-safe кутовими residuals (arctan2(sin, cos)).
"""

import json
from collections import defaultdict

import faiss
import h5py
import numpy as np

from config import get_cfg
from src.geometry.affine_utils import (
    compose_affine_5dof,
    decompose_affine,
    decompose_affine_5dof,
    unwrap_angles,
)
from src.geometry.pose_graph.vo_guards import (
    check_anchor_gaps,
    downweight_gap_edges,
    select_gap_fallback_frames,
    temporal_edge_sane,
)
from src.geometry.pose_graph_optimizer import (
    PoseGraphOptimizer,
    affine_fit_residual,
    homography_to_similarity,
)
from src.geometry.transformations import GeometryTransforms
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PropagationPipeline:
    """
    Графова пропагація з глобальною оптимізацією.

    Фази:
      1. Prefetch фіч → побудова часових ребер (sequential matching)
      2. Loop closure detection (FAISS DINOv2 retrieval → LightGlue matching)
      3. Фіксація GPS-якорів + BFS ініціалізація початкового наближення
      4. Глобальна оптимізація (Levenberg-Marquardt)
      5. Збереження результатів у HDF5
    """

    def __init__(
        self,
        database,
        calibration,
        matcher,
        config=None,
        progress_callback=None,
        error_callback=None,
        completed_callback=None,
    ):
        # Qt-free ядро графової пропагації. progress/error/completed
        # виходять через колбеки (CalibrationPropagationWorker під'єднує
        # їх до однойменних Qt-сигналів).
        self._progress_cb = progress_callback
        self._error_cb = error_callback
        self._completed_cb = completed_callback
        self.database = database
        self.calibration = calibration
        self.matcher = matcher
        self.config = config or {}
        self._is_running = True

        self.min_matches = get_cfg(self.config, "localization.min_matches", 15)
        self.ransac_thresh = get_cfg(self.config, "localization.ransac_threshold", 3.0)
        self.homography_backend = get_cfg(self.config, "homography.backend", "opencv")
        self.use_mad_ransac = get_cfg(self.config, "homography.use_mad_ransac", True)
        self.mad_k_factor = get_cfg(self.config, "homography.mad_k_factor", 2.5)

        self.frame_w = self.database.metadata.get("frame_width", 1920)
        self.frame_h = self.database.metadata.get("frame_height", 1080)

        # Параметри графової оптимізації
        self.lc_top_k = get_cfg(self.config, "graph_optimization.loop_closure_top_k", 5)
        self.lc_min_sim = get_cfg(
            self.config, "graph_optimization.loop_closure_min_similarity", 0.75
        )
        self.lc_min_gap = get_cfg(self.config, "graph_optimization.loop_closure_min_frame_gap", 3)
        self.lc_auto_min_gap = get_cfg(
            self.config, "graph_optimization.loop_closure_auto_min_gap", False
        )
        self.lc_overlap_factor = get_cfg(
            self.config, "graph_optimization.loop_closure_overlap_factor", 1.0
        )
        self.lc_dist_prefilter = get_cfg(
            self.config, "graph_optimization.loop_closure_dist_prefilter", False
        )
        self.lc_dist_margin = get_cfg(
            self.config, "graph_optimization.loop_closure_dist_margin", 2.0
        )
        self.lc_odometry_check = get_cfg(
            self.config, "graph_optimization.loop_closure_odometry_check", False
        )
        self.odometry_margin = get_cfg(
            self.config, "graph_optimization.odometry_consistency_margin", 1.5
        )
        self.odometry_drift_frac = get_cfg(
            self.config, "graph_optimization.odometry_drift_frac", 0.25
        )
        self.odometry_inconsistency_factor = get_cfg(
            self.config, "graph_optimization.odometry_inconsistency_factor", 0.3
        )
        self._prelim_states: dict[int, object] = {}
        self._prelim_centers: dict[int, object] = {}
        self._prelim_dist_threshold = 0.0
        self.lc_min_inliers = get_cfg(
            self.config, "graph_optimization.loop_closure_min_inliers", 15
        )
        self.temporal_base_w = get_cfg(
            self.config, "graph_optimization.temporal_edge_base_weight", 1.0
        )
        self.spatial_base_w = get_cfg(
            self.config, "graph_optimization.spatial_edge_base_weight", 2.0
        )
        self.max_iters = get_cfg(self.config, "graph_optimization.max_iterations", 50)
        self.tolerance = get_cfg(self.config, "graph_optimization.convergence_tolerance", 1e-6)
        self.use_bfs = get_cfg(self.config, "graph_optimization.use_bfs_initialization", True)
        self.export_geojson = get_cfg(self.config, "graph_optimization.export_geojson", True)

        # Скільки кадрів можна "перестрибнути" при побудові temporal ребер
        self.max_skip_frames = get_cfg(self.config, "propagation.max_skip_frames", 3)
        self.rotation_retry = get_cfg(self.config, "propagation.rotation_retry", False)
        self.temporal_weight_use_fit = get_cfg(
            self.config, "graph_optimization.temporal_weight_use_fit_quality", False
        )
        self.temporal_fit_k = get_cfg(
            self.config, "graph_optimization.temporal_fit_quality_k", 0.05
        )

        # ── Нові опції (Етапи 2/3/4). Дефолти off = поточна поведінка. ──
        go = "graph_optimization."
        self.use_analytic_jac = get_cfg(self.config, go + "use_analytic_jacobian", False)
        self.warm_start = get_cfg(self.config, go + "warm_start", False)
        self.two_stage_prune = get_cfg(self.config, go + "two_stage_prune", False)
        self.prune_mad_k = get_cfg(self.config, go + "prune_mad_k", 5.0)
        self.prune_max_spatial_frac = get_cfg(self.config, go + "prune_max_spatial_frac", 0.2)
        self.gnc_spatial = get_cfg(self.config, go + "gnc_spatial", False)
        self.gnc_rounds = get_cfg(self.config, go + "gnc_rounds", 5)
        self.gnc_mad_k = get_cfg(self.config, go + "gnc_mad_k", 3.0)
        self.kinematic_prior_weight = get_cfg(
            self.config, go + "kinematic_prior_weight", 0.0
        )
        self.pchip_gap_fill = get_cfg(self.config, go + "pchip_gap_fill", False)
        self.log_scale_interp = get_cfg(self.config, go + "log_scale_interp", False)
        self.edge_gate_enabled = get_cfg(self.config, go + "edge_gate_enabled", False)
        self.edge_gate_max_rot = get_cfg(self.config, go + "edge_gate_max_rotation_deg", 40.0)
        self.edge_gate_max_scale = get_cfg(self.config, go + "edge_gate_max_scale_ratio", 1.6)
        self.edge_gate_min_inlier_ratio = get_cfg(
            self.config, go + "edge_gate_min_inlier_ratio", 0.25
        )
        self.edge_gate_mutual = get_cfg(self.config, go + "edge_gate_mutual_check", True)
        self.edge_gate_cluster = get_cfg(self.config, go + "edge_gate_cluster_consistency", True)
        self.spatial_weight_use_sim = get_cfg(
            self.config, go + "spatial_weight_use_similarity", False
        )

        # М'які якорі (Етап 1.1). off = fix_node (жорсткий, поточна поведінка).
        self.soft_anchors = get_cfg(self.config, go + "soft_anchors", False)
        self.anchor_base_w = get_cfg(self.config, go + "anchor_base_w", 200.0)
        self.anchor_sigma_floor_m = get_cfg(self.config, go + "anchor_sigma_floor_m", 0.05)
        self.anchor_loo_threshold_m = get_cfg(self.config, go + "anchor_loo_threshold_m", 5.0)

        # ── Етап 8 (сесія 2026-07-12): запобіжники temporal-VO. Дефолти off. ──
        self.temporal_edge_gate = get_cfg(self.config, go + "temporal_edge_gate", False)
        self.temporal_gate_max_rot = get_cfg(
            self.config, go + "temporal_gate_max_rotation_deg", 30.0
        )
        self.temporal_gate_max_scale = get_cfg(
            self.config, go + "temporal_gate_max_scale_ratio", 1.4
        )
        self.temporal_gate_max_shift_frac = get_cfg(
            self.config, go + "temporal_gate_max_shift_frac", 1.2
        )
        self.anchor_gap_check = get_cfg(self.config, go + "anchor_gap_check", False)
        self.anchor_gap_max_dev_m = get_cfg(self.config, go + "anchor_gap_max_dev_m", 150.0)
        self.anchor_gap_downweight = get_cfg(self.config, go + "anchor_gap_downweight", 0.05)
        self.skip_bridges = get_cfg(self.config, "propagation.skip_bridges", False)
        self.mnn_fallback = get_cfg(self.config, "propagation.mnn_fallback", False)
        self._n_rotation_retry = 0
        self._origin_xy = (0.0, 0.0)

    def stop(self):
        self._is_running = False

    def _report_progress(self, pct, msg):
        if self._progress_cb is not None:
            self._progress_cb(pct, msg)

    def _report_error(self, msg):
        if self._error_cb is not None:
            self._error_cb(msg)

    def _report_completed(self):
        if self._completed_cb is not None:
            self._completed_cb()

    # ─── Головний метод ──────────────────────────────────────────────────────

    def _propagate(self):
        num_frames = self.database.get_num_frames()
        all_anchors = sorted(self.calibration.anchors, key=lambda a: a.frame_id)
        anchors = [a for a in all_anchors if a.frame_id < num_frames]

        # ВИПРАВЛЕНО: раніше якорі поза межами БД викидалися МОВЧКИ (тільки лог),
        # і користувач не знав, що половина його якорів не використовується.
        dropped = [a.frame_id for a in all_anchors if a.frame_id >= num_frames]
        if dropped:
            self._report_error(
                f"Якорі для кадрів {dropped} виходять за межі бази даних "
                f"({num_frames} слотів). Ймовірно, вони були створені за номерами "
                f"кадрів оригінального відео, а не слотів БД (кадр_відео // frame_step). "
                f"Видаліть ці якорі та додайте заново через діалог калібрування."
            )
            return

        if not anchors:
            self._report_error("Немає якорів калібрування")
            return

        logger.info(
            f"Starting GRAPH propagation for {num_frames} frames "
            f"using {len(anchors)} anchors: "
            f"{[f'#{a.frame_id}' for a in anchors]}"
        )

        # ── Phase 1: Prefetch + Temporal edges ───────────────────────────────
        self._report_progress(0, "Передзавантаження фіч у RAM...")
        all_features = self._prefetch_features(num_frames)

        optimizer = PoseGraphOptimizer(self.frame_w, self.frame_h)
        for i in range(num_frames):
            if i in all_features:
                optimizer.add_node(i)

        self._report_progress(10, "Побудова часових ребер (sequential matching)...")
        temporal_count = self._build_temporal_edges(optimizer, all_features, num_frames)
        logger.info(f"Phase 1 complete: {temporal_count} temporal edges")

        # Авто min_frame_gap (Етап 2.1): з медіанного руху за слот. Замінює ручну
        # константу; працює в парі з odometry-consistency (2.3), що ловить
        # аліасні same-leg замикання в зоні поза фізичним перекриттям.
        if self.lc_auto_min_gap:
            auto_gap = optimizer.estimate_min_loop_gap(
                self.frame_w, self.frame_h, self.lc_overlap_factor
            )
            if auto_gap is not None:
                logger.info(f"Auto min_frame_gap: {auto_gap} слотів (було {self.lc_min_gap})")
                self.lc_min_gap = auto_gap

        # Дистанційний префільтр (Етап 2.2): прикидка центрів BFS-ланцюгом temporal
        # від якорів → поріг margin×діагональ_кадру у метрах. Далекі пари не матчаться.
        self._prelim_states = {}
        self._prelim_centers = {}
        self._prelim_dist_threshold = 0.0
        if self.lc_dist_prefilter or self.lc_odometry_check:
            seed_affines = {
                a.frame_id: a.affine_matrix for a in anchors if a.frame_id in all_features
            }
            self._prelim_states = optimizer.preliminary_states(seed_affines)
            self._prelim_centers = {fid: st[:2] for fid, st in self._prelim_states.items()}
            if self.lc_dist_prefilter and self._prelim_centers:
                scales = [
                    float(np.sqrt(abs(np.linalg.det(np.asarray(a.affine_matrix)[:2, :2]))))
                    for a in anchors
                ]
                scale_m_per_px = float(np.median(scales)) if scales else 0.0
                frame_diag_px = float(np.hypot(self.frame_w, self.frame_h))
                self._prelim_dist_threshold = self.lc_dist_margin * frame_diag_px * scale_m_per_px
                logger.info(
                    f"Dist prefilter: поріг {self._prelim_dist_threshold:.1f} м, "
                    f"{len(self._prelim_centers)} прикидок центрів"
                )

        # ── Phase 2: Loop closure detection ──────────────────────────────────
        self._report_progress(30, "Пошук просторових замикань (loop closure)...")
        spatial_count = self._detect_loop_closures(optimizer, all_features, num_frames)
        logger.info(f"Phase 2 complete: {spatial_count} spatial edges (loop closures)")
        logger.info(
            f"Graph: {optimizer.num_nodes} nodes, {optimizer.num_edges} edges "
            f"({temporal_count} temporal + {spatial_count} spatial)"
        )

        # ── Phase 3: Fix anchors (Local Origin Strategy) ──────────────────────
        self._report_progress(60, "Фіксація GPS-якорів (Local Origin)...")

        # ВИПРАВЛЕНО: якір міг потрапити на порожній слот (keyframe selection
        # пропустила кадр) — тоді fix_node створював ізольований вузол без ребер,
        # і якір мовчки ігнорувався оптимізацією. Снапимо до найближчого кадру
        # з фічами: рух між сусідніми слотами в такому гепі нижчий за пороги
        # keyframe selection, тому похибка снапу мізерна.
        feature_ids = np.array(sorted(all_features.keys()), dtype=np.int64)
        if len(feature_ids) == 0:
            self._report_error("У базі даних немає жодного кадру з фічами")
            return

        anchor_nodes: dict[int, object] = {}
        for anchor in anchors:
            fid = anchor.frame_id
            if fid not in all_features:
                nearest = int(feature_ids[np.argmin(np.abs(feature_ids - fid))])
                logger.warning(
                    f"Anchor frame {fid} has no features (non-keyframe slot). "
                    f"Snapping to nearest keyframe {nearest} (Δ={abs(nearest - fid)} slots)."
                )
                fid = nearest
            if fid in anchor_nodes:
                logger.warning(
                    f"Two anchors mapped to the same node {fid} after snapping; "
                    f"keeping the first one."
                )
                continue
            anchor_nodes[fid] = anchor

        # Визначаємо локальну опорну точку для математичної стабільності (Local Center)
        # Використовуємо метричну трансляцію першого якоря
        ref_anchor = anchors[0]
        origin_tx = float(ref_anchor.affine_matrix[0, 2])
        origin_ty = float(ref_anchor.affine_matrix[1, 2])
        logger.info(f"Local Origin established at: ({origin_tx:.2f}, {origin_ty:.2f})")
        self._origin_xy = (origin_tx, origin_ty)

        for fid, anchor in anchor_nodes.items():
            # Створюємо копію матриці з відносною трансляцією
            local_affine = anchor.affine_matrix.copy().astype(np.float64)
            local_affine[0, 2] -= origin_tx
            local_affine[1, 2] -= origin_ty
            if self.soft_anchors:
                # σ = rmse_m якоря: GT (≈0)→floor→жорсткий; реальний (5–10 м)→м'який
                optimizer.add_anchor(
                    fid,
                    local_affine,
                    sigma_m=float(getattr(anchor, "rmse_m", 0.0)),
                    base_w=self.anchor_base_w,
                    sigma_floor=self.anchor_sigma_floor_m,
                )
            else:
                optimizer.fix_node(fid, local_affine)

        # ── Етап 8.2: звірка проміжків між якорями ДО оптимізації ────────────
        # Консистентний аліасинг (усі ребра проміжку брешуть однаково) невидимий
        # для резидуалів; неузгоджені проміжки глушаться, їх кадри після
        # оптимізації перезаповнюються інтерполяцією по якорях.
        flagged_gaps: list[tuple[int, int]] = []
        gap_report: dict = {}
        if self.anchor_gap_check:
            gap_report = check_anchor_gaps(
                optimizer.edges,
                optimizer.anchor_states(),
                optimizer.sign,
                self.anchor_gap_max_dev_m,
            )
            flagged_gaps = [k for k, v in gap_report.items() if v["status"] != "ok"]
            for a, b in flagged_gaps:
                v = gap_report[(a, b)]
                dev = (
                    f"розбіжність {v['dev_m']:.0f} м"
                    if v["dev_m"] is not None
                    else "ланцюг розірваний"
                )
                logger.warning(
                    f"Проміжок якорів #{a}→#{b} не узгоджений із VO-ланцюгом ({dev}) — "
                    f"кадри проміжку підуть на інтерполяцію по якорях"
                )
            n_dw = downweight_gap_edges(
                optimizer.edges,
                [k for k in flagged_gaps if gap_report[k]["status"] == "inconsistent"],
                self.anchor_gap_downweight,
            )
            if n_dw:
                logger.info(
                    f"Етап 8.2: приглушено {n_dw} temporal-ребер (вага ×{self.anchor_gap_downweight})"
                )

        # Warm start (Етап 4.2): x0 з попереднього розв'язку замість BFS з нуля.
        # BFS нижче лишається для першого запуску та як fallback (заповнює лише
        # вузли БЕЗ стану).
        if self.warm_start:
            prev = self._load_previous_affines()
            if prev:
                local_prev = {}
                for fid, aff in prev.items():
                    a = np.asarray(aff, dtype=np.float64).copy()
                    a[0, 2] -= origin_tx
                    a[1, 2] -= origin_ty
                    local_prev[fid] = a
                optimizer.warm_start_from_affines(local_prev)

        if self.use_bfs:
            bfs_count = optimizer.initialize_from_bfs()
            logger.info(f"Phase 3 complete: {bfs_count} nodes initialized via BFS")
        else:
            logger.info("Phase 3 complete: BFS initialization skipped (disabled)")

        # ── Phase 4: Optimize ────────────────────────────────────────────────
        self._report_progress(70, "Глобальна оптимізація графу (Levenberg-Marquardt)...")
        results = optimizer.optimize(
            max_iterations=self.max_iters,
            tolerance=self.tolerance,
            progress_callback=lambda msg: self._report_progress(70, msg),
            use_analytic_jac=self.use_analytic_jac,
            two_stage_prune=self.two_stage_prune,
            prune_mad_k=self.prune_mad_k,
            prune_max_spatial_frac=self.prune_max_spatial_frac,
            gnc_spatial=self.gnc_spatial,
            gnc_rounds=self.gnc_rounds,
            gnc_mad_k=self.gnc_mad_k,
            kinematic_prior_weight=self.kinematic_prior_weight,
        )
        logger.info(f"Phase 4 complete: {len(results)} frames optimized")

        # Звіт пропагації (Етап 1.3): класи ребер, резидуали, топ-гірших, anchor stress
        try:
            logger.info(
                "Звіт пропагації:\n"
                + optimizer.format_diagnostics(loo_threshold_m=self.anchor_loo_threshold_m)
            )
        except Exception as diag_err:
            logger.warning(f"Diagnostics report failed: {diag_err}")

        # ── Етап 8.2: кадри неузгоджених проміжків → на інтерполяцію по якорях.
        # Рахуємо в ЛОКАЛЬНИХ координатах (до відновлення origin).
        force_invalid: set[int] = set()
        if self.anchor_gap_check and flagged_gaps:
            cxp, cyp = self.frame_w / 2.0, self.frame_h / 2.0
            centers = {
                fid: (
                    float(aff[0, 0] * cxp + aff[0, 1] * cyp + aff[0, 2]),
                    float(aff[1, 0] * cxp + aff[1, 1] * cyp + aff[1, 2]),
                )
                for fid, aff in results.items()
            }
            force_invalid = select_gap_fallback_frames(
                centers, optimizer.anchor_states(), flagged_gaps, self.anchor_gap_max_dev_m
            )
            if force_invalid:
                logger.info(
                    f"Етап 8.2: {len(force_invalid)} кадрів перезаповнюються інтерполяцією "
                    f"(відхилення від лінії якорів > {self.anchor_gap_max_dev_m:.0f} м): "
                    f"{sorted(force_invalid)}"
                )

        # Відновлюємо абсолютні координати (додаємо Local Origin назад)
        for fid in results:
            results[fid][0, 2] += origin_tx
            results[fid][1, 2] += origin_ty

        # ── Phase 5: Save to HDF5 ───────────────────────────────────────────
        self._report_progress(85, "Збереження результатів у HDF5...")
        valid_count = self._save_to_hdf5(results, anchors, optimizer, force_invalid=force_invalid)

        # Експорт GeoJSON для візуалізації
        if self.export_geojson and self.calibration.converter:
            try:
                geojson = optimizer.export_graph_geojson(
                    self.calibration.converter,
                    self.frame_w,
                    self.frame_h,
                    origin_xy=self._origin_xy,
                )
                geojson_path = str(self.database.db_path).replace(".h5", "_graph.geojson")
                with open(geojson_path, "w", encoding="utf-8") as f:
                    json.dump(geojson, f, indent=2, ensure_ascii=False)
                logger.success(f"Graph exported to GeoJSON: {geojson_path}")
            except Exception as e:
                logger.warning(f"GeoJSON export failed: {e}")

        self._report_progress(
            100,
            f"Готово! {valid_count}/{num_frames} кадрів отримали координати "
            f"({temporal_count} часових + {spatial_count} просторових ребер).",
        )
        self._report_completed()

    # ─── Phase 1: Prefetch + Temporal edges ──────────────────────────────────

    def _prefetch_features(self, num_frames: int) -> dict:
        """Завантажує всі фічі в RAM."""
        features = {}
        for i in range(num_frames):
            if not self._is_running:
                return features
            try:
                features[i] = self.database.get_local_features(i)
            except Exception:
                pass
            if i % 500 == 0:
                self._report_progress(
                    int(i / num_frames * 8),
                    f"Prefetch: {i}/{num_frames}",
                )
        logger.info(f"Prefetched features for {len(features)} frames")
        return features

    def _build_temporal_edges(
        self,
        optimizer: PoseGraphOptimizer,
        features: dict,
        num_frames: int,
    ) -> int:
        """Побудова часових ребер між послідовними кадрами."""
        count = 0
        self._n_rotation_retry = 0
        n_gated = 0
        n_bridged = 0
        # ВИПРАВЛЕНО (раніше): ребро будується до попереднього кадру-З-ФІЧАМИ,
        # незалежно від гепа keyframe selection. Етап 8 (2026-07-12): опційно
        # (а) санітарний гейт трансформації ребра, (б) мости через розриви —
        # якщо матч із найближчим сусідом упав/відсіяний, пробуємо глибших
        # (до max_skip_frames), щоб ланцюг не розпадався на «острови» та
        # «апендикси» без другого якоря (кадри 1–21 lasttest → відліт на км).
        recent: list[tuple[int, dict]] = []

        for i in range(num_frames):
            if not self._is_running:
                break

            feat_i = features.get(i)
            if feat_i is None:
                continue

            if recent:
                gap = i - recent[-1][0]
                if gap > self.max_skip_frames:
                    logger.debug(
                        f"Temporal edge across keyframe gap: {recent[-1][0]} → {i} ({gap} slots)"
                    )
                candidates = recent[::-1] if self.skip_bridges else recent[-1:]
                for depth, (last_id, last_feat) in enumerate(candidates):
                    similarity, inliers, rmse_val, result = self._try_temporal_pair(
                        feat_i, last_feat, last_id, i
                    )
                    if similarity is None:
                        continue
                    if self.temporal_edge_gate:
                        ok, reason = temporal_edge_sane(
                            similarity,
                            i - last_id,
                            self.frame_w,
                            self.frame_h,
                            self.temporal_gate_max_rot,
                            self.temporal_gate_max_scale,
                            self.temporal_gate_max_shift_frac,
                        )
                        if not ok:
                            n_gated += 1
                            logger.debug(f"Temporal gate відсіяв {last_id}→{i}: {reason}")
                            continue
                    weight = self._compute_weight(inliers, rmse_val, self.temporal_base_w)
                    # 6.2: менша довіра кадрам із нахилом/рельєфом (великий залишок
                    # афінного фіту H). Лише первинний матч (result), не rotation-retry.
                    if self.temporal_weight_use_fit and result is not None:
                        fit_res = affine_fit_residual(result[0], self.frame_w, self.frame_h)
                        if fit_res is not None:
                            weight *= 1.0 / (1.0 + self.temporal_fit_k * fit_res)
                    optimizer.add_edge(
                        from_id=last_id,
                        to_id=i,
                        relative_affine_2x3=similarity,
                        weight=weight,
                        edge_type="temporal",
                        inliers=inliers,
                        rmse=rmse_val,
                    )
                    count += 1
                    if depth > 0:
                        n_bridged += 1
                    break

            recent.append((i, feat_i))
            depth_limit = max(1, self.max_skip_frames) if self.skip_bridges else 1
            if len(recent) > depth_limit:
                recent.pop(0)

            if i % 200 == 0:
                self._report_progress(
                    10 + int(i / num_frames * 18),
                    f"Часові ребра: {count} (кадр {i}/{num_frames})",
                )

        if self.rotation_retry and self._n_rotation_retry:
            logger.info(f"Rotation-retry врятував {self._n_rotation_retry} temporal-ребер")
        if n_gated:
            logger.info(
                f"Temporal gate відсіяв {n_gated} ребер (дегенеративні трансформації)"
            )
        if n_bridged:
            logger.info(f"Skip-мости з'єднали {n_bridged} розривів temporal-ланцюга")
        return count

    def _try_temporal_pair(self, feat_i, last_feat, last_id, i):
        """Одна спроба temporal-матчу пари (last_id → i): основний матч +
        (за прапорцем) rotation-retry (Етап 5). Повертає
        (similarity | None, inliers, rmse, result_or_None)."""
        similarity = None
        inliers = 0
        rmse_val = 0.0
        # H maps feat_i → last_feat (to→from direction)
        result = self._match_and_build_edge(feat_i, last_feat)
        if result is not None:
            H, inliers, rmse_val, _n_matches = result
            similarity = homography_to_similarity(H, self.frame_w, self.frame_h)

        # Ротаційна робастність (Етап 5): матч упав → пробуємо з поворотом
        # query на кут ланцюга frame_poses, далі перебір k·90°.
        if similarity is None and self.rotation_retry:
            retry = self._temporal_rotation_retry(feat_i, last_feat, last_id, i)
            if retry is not None:
                similarity, inliers, rmse_val = retry
                self._n_rotation_retry += 1

        return similarity, inliers, rmse_val, result

    def _temporal_rotation_retry(self, feat_i, last_feat, from_id, to_id):
        """Повторний temporal-матч із поворотом query (Етап 5). Повертає
        (similarity, inliers, rmse) або None. Кут — із ланцюга frame_poses БД
        (одометричний пріор), fallback — перебір k·90°. Отриману H_r
        (rotated_query→ref) компонуємо назад: H_true = H_r · R(θ)."""
        from src.localization.rotation_geometry import (
            chain_relative_angle_deg,
            rotate_keypoints,
            rotation_homography,
            temporal_retry_angles,
        )

        cx, cy = self.frame_w / 2.0, self.frame_h / 2.0
        chain_angle = None
        fp = getattr(self.database, "frame_poses", None)
        if fp is not None and 0 <= to_id < len(fp) and 0 <= from_id < len(fp):
            # to→from: кут, яким повертаємо query(to), щоб вирівняти з ref(from)
            chain_angle = chain_relative_angle_deg(fp[to_id], fp[from_id])

        for ang_deg in temporal_retry_angles(chain_angle, use_chain=chain_angle is not None):
            ang = np.radians(ang_deg)
            feat_rot = dict(feat_i)
            feat_rot["keypoints"] = rotate_keypoints(feat_i["keypoints"], ang, cx, cy)
            result = self._match_and_build_edge(feat_rot, last_feat)
            if result is None:
                continue
            H_r, inliers, rmse_val, _n = result
            H_true = np.asarray(H_r, dtype=np.float64) @ rotation_homography(ang, cx, cy)
            similarity = homography_to_similarity(H_true, self.frame_w, self.frame_h)
            if similarity is not None:
                logger.debug(
                    f"Rotation-retry {from_id}→{to_id} OK @ {ang_deg:.1f}° (chain={chain_angle})"
                )
                return similarity, inliers, rmse_val
        return None

    # ─── Phase 2: Loop closure detection ─────────────────────────────────────

    def _detect_loop_closures(
        self,
        optimizer: PoseGraphOptimizer,
        features: dict,
        num_frames: int,
    ) -> int:
        """Знаходить просторові замикання через DINOv2 (LanceDB/FAISS) + LightGlue matching."""

        has_lancedb = (
            hasattr(self.database, "lance_table") and self.database.lance_table is not None
        )
        lance_table = self.database.lance_table if has_lancedb else None

        # Завантажуємо вектори для швидкого пошуку
        global_desc_dict = {}

        if has_lancedb:
            logger.info("Extracting global descriptors from LanceDB for loop closures...")
            try:
                df = lance_table.to_pandas()
                for _, row in df.iterrows():
                    fid = int(row["frame_id"])
                    global_desc_dict[fid] = np.array(row["vector"], dtype=np.float32)
            except Exception as e:
                logger.warning(f"Failed to load vectors from LanceDB: {e}")
        else:
            global_desc = self.database.global_descriptors
            if global_desc is not None and len(global_desc) > 0:
                for i in range(len(global_desc)):
                    if np.any(global_desc[i]):
                        global_desc_dict[i] = global_desc[i]

        if not global_desc_dict:
            logger.warning("No global descriptors available — skipping loop closure detection")
            return 0

        # Нормалізація векторів
        normed_dict = {}
        for fid, vec in global_desc_dict.items():
            norm = np.linalg.norm(vec)
            normed_dict[fid] = vec / (norm + 1e-8) if norm > 0 else vec

        # Побудова FAISS індексу якщо немає LanceDB
        faiss_index = None
        faiss_id_map = []
        if not has_lancedb:
            dim = next(iter(normed_dict.values())).shape[0]
            faiss_index = faiss.IndexFlatIP(dim)
            mat = []
            for fid, vec in normed_dict.items():
                mat.append(vec)
                faiss_id_map.append(fid)
            faiss_index.add(np.array(mat, dtype=np.float32))
            logger.info(f"FAISS index built: {faiss_index.ntotal} vectors, dim={dim}")

        # ── Pass 1: retrieval top-k для всіх кадрів (для взаємної перевірки 2.2) ──
        retrieval: dict[int, list[tuple[int, float]]] = {}
        for i in range(num_frames):
            if not self._is_running:
                break
            feat_i = features.get(i)
            if feat_i is None or i not in normed_dict:
                continue
            retrieval[i] = self._retrieve_candidates(
                normed_dict[i], has_lancedb, lance_table, faiss_index, faiss_id_map
            )
        topk_sets = {fid: {int(j) for j, _ in cands} for fid, cands in retrieval.items()}

        # ── Pass 2: збір spatial-кандидатів + гейти (Етап 2) ──
        already_matched: set[tuple[int, int]] = set()
        specs: list[dict] = []
        n_gated_phys = 0
        n_gated_mutual = 0
        n_gated_dist = 0

        for i in range(num_frames):
            if not self._is_running:
                break
            feat_i = features.get(i)
            if feat_i is None or i not in retrieval:
                continue

            for j, sim_score in retrieval[i]:
                j = int(j)
                if j == i or j == -1:
                    continue
                if abs(i - j) <= self.lc_min_gap:
                    continue
                if float(sim_score) < self.lc_min_sim:
                    continue

                edge_key = (min(i, j), max(i, j))
                if edge_key in already_matched:
                    continue
                already_matched.add(edge_key)

                # 2.2 взаємність retrieval: j теж має бачити i у своєму top-k
                if self.edge_gate_enabled and self.edge_gate_mutual:
                    if i not in topk_sets.get(j, ()):
                        n_gated_mutual += 1
                        continue

                # 2.2 дистанційний префільтр: якщо прикидки центрів далеко — не матчимо
                if self.lc_dist_prefilter and self._prelim_dist_threshold > 0:
                    ci = self._prelim_centers.get(i)
                    cj = self._prelim_centers.get(j)
                    if (
                        ci is not None
                        and cj is not None
                        and (
                            float(np.linalg.norm(np.asarray(ci) - np.asarray(cj)))
                            > self._prelim_dist_threshold
                        )
                    ):
                        n_gated_dist += 1
                        continue

                feat_j = features.get(j)
                if feat_j is None:
                    continue

                # Matching: feat_j → feat_i (H maps j pixels → i pixels = to→from)
                result = self._match_and_build_edge(feat_j, feat_i)
                if result is None:
                    continue

                H, inliers, rmse_val, n_matches = result
                if inliers < self.lc_min_inliers:
                    continue

                similarity = homography_to_similarity(H, self.frame_w, self.frame_h)
                if similarity is None:
                    continue

                # 2.1 фізичні межі відносної трансформації
                if self.edge_gate_enabled and not self._passes_physical_gate(
                    similarity, inliers, n_matches
                ):
                    n_gated_phys += 1
                    continue

                specs.append(
                    {
                        "i": i,
                        "j": j,
                        "similarity": similarity,
                        "inliers": inliers,
                        "rmse": rmse_val,
                        "sim": float(sim_score),
                    }
                )

            if i % 200 == 0:
                self._report_progress(
                    30 + int(i / num_frames * 28),
                    f"Loop closure: {len(specs)} кандидатів (кадр {i}/{num_frames})",
                )

        # ── Pass 3: кластерна узгодженість (2.3) + фінальні ваги + додавання ──
        cluster_factor = (
            self._cluster_consistency_factors(specs)
            if (self.edge_gate_enabled and self.edge_gate_cluster)
            else None
        )
        # 2.3 odometry-consistency: несумісні з temporal-ланцюгом spatial-ребра → ×factor
        odometry_factor = (
            optimizer.odometry_consistency_factors(
                specs,
                self._prelim_states,
                self.frame_w,
                self.frame_h,
                margin=self.odometry_margin,
                drift_frac=self.odometry_drift_frac,
                factor=self.odometry_inconsistency_factor,
            )
            if (self.lc_odometry_check and self._prelim_states)
            else None
        )
        n_odo_down = sum(1 for f in odometry_factor if f < 1.0) if odometry_factor else 0
        for idx, spec in enumerate(specs):
            weight = self._compute_weight(spec["inliers"], spec["rmse"], self.spatial_base_w)
            if self.spatial_weight_use_sim:  # 4.3: w *= 0.5 + 0.5·sim
                weight *= 0.5 + 0.5 * max(0.0, min(1.0, spec["sim"]))
            if cluster_factor is not None:  # 2.3 (cluster): самотнє ребро → ×0.5
                weight *= cluster_factor[idx]
            if odometry_factor is not None:  # 2.3 (odometry): несумісне ребро → ×factor
                weight *= odometry_factor[idx]
            optimizer.add_edge(
                from_id=spec["i"],
                to_id=spec["j"],
                relative_affine_2x3=spec["similarity"],
                weight=weight,
                edge_type="spatial",
                inliers=spec["inliers"],
                rmse=spec["rmse"],
            )

        if self.edge_gate_enabled or self.lc_dist_prefilter or self.lc_odometry_check:
            logger.info(
                f"Edge gating: {n_gated_phys} відсіяно фізично, "
                f"{n_gated_mutual} за взаємністю retrieval, "
                f"{n_gated_dist} дистанційним префільтром; "
                f"{n_odo_down} ребер ×odometry-factor (несумісні з ланцюгом)"
            )
        return len(specs)

    # ─── Гейти ребер (Етап 2) ────────────────────────────────────────────────

    def _retrieve_candidates(
        self, q, has_lancedb, lance_table, faiss_index, faiss_id_map
    ) -> list[tuple[int, float]]:
        """top-k схожих кадрів (id, similarity). Логіка ідентична попередній."""
        candidates: list[tuple[int, float]] = []
        if has_lancedb:
            try:
                res = (
                    lance_table.search(q.astype(np.float32))
                    .metric("cosine")
                    .limit(self.lc_top_k + 1)
                    .select(["frame_id", "_distance"])
                    .to_list()
                )
                for r in res:
                    candidates.append((int(r["frame_id"]), max(0.0, 1.0 - r["_distance"])))
            except Exception:
                pass
        else:
            q_batch = np.array([q], dtype=np.float32)
            scores, ids = faiss_index.search(q_batch, self.lc_top_k + 1)
            for raw_idx, sim_score in zip(ids[0], scores[0]):
                if raw_idx != -1:
                    candidates.append((int(faiss_id_map[raw_idx]), float(sim_score)))
        return candidates

    def _passes_physical_gate(self, similarity, inliers: int, n_matches: int) -> bool:
        """Фізичні межі відносної трансформації хибного loop closure (2.1)."""
        _, _, sx, sy, angle = decompose_affine_5dof(similarity)
        if abs(np.degrees(angle)) > self.edge_gate_max_rot:
            return False
        max_log = np.log(max(self.edge_gate_max_scale, 1.0 + 1e-9))
        if abs(np.log(max(sx, 1e-9))) > max_log or abs(np.log(max(sy, 1e-9))) > max_log:
            return False
        if n_matches > 0 and (inliers / n_matches) < self.edge_gate_min_inlier_ratio:
            return False
        return True

    def _cluster_consistency_factors(self, specs: list[dict], window: int = 3) -> list[float]:
        """Самотнє loop closure без сусіда з близькими кінцями → вага ×0.5 (2.3)."""
        factors = [1.0] * len(specs)
        for a in range(len(specs)):
            ia, ja = specs[a]["i"], specs[a]["j"]
            supported = False
            for b in range(len(specs)):
                if b == a:
                    continue
                ib, jb = specs[b]["i"], specs[b]["j"]
                d1 = abs(ia - ib) + abs(ja - jb)
                d2 = abs(ia - jb) + abs(ja - ib)
                if min(d1, d2) <= 2 * window:
                    supported = True
                    break
            if not supported:
                factors[a] = 0.5
        return factors

    # ─── Phase 5: Save to HDF5 ───────────────────────────────────────────────

    def _save_to_hdf5(
        self,
        results: dict[int, np.ndarray],
        anchors,
        optimizer: PoseGraphOptimizer,
        force_invalid: set[int] | None = None,
    ) -> int:
        """Зберігає оптимізовані афінні матриці у HDF5.

        Формат 100% сумісний з існуючим DatabaseLoader.
        """
        num_frames = self.database.get_num_frames()
        frame_affine = np.zeros((num_frames, 2, 3), dtype=np.float64)
        frame_valid = np.zeros(num_frames, dtype=bool)
        frame_rmse = np.zeros(num_frames, dtype=np.float64)
        frame_disagreement = np.zeros(num_frames, dtype=np.float64)
        frame_matches = np.zeros(num_frames, dtype=np.int32)

        # Записуємо результати оптимізації
        # Оскільки optimizer повертає ТІЛЬКИ досяжні вузли,
        # незв'язані кадри залишаться з frame_valid = False
        # Етап 8.2: кадри неузгоджених проміжків пропускаємо — їх заповнить
        # штатна інтерполяція (pchip/лінійна) по якорях і валідних сусідах.
        skip = force_invalid or set()
        for frame_id, affine in results.items():
            if 0 <= frame_id < num_frames and frame_id not in skip:
                frame_affine[frame_id] = affine.astype(np.float64)
                frame_valid[frame_id] = True

        filled_count = self._fill_gaps_by_interpolation(frame_affine, frame_valid)
        if filled_count > 0:
            logger.info(f"Interpolated coordinates for {filled_count} missing frames")

        # Обчислюємо QA метрики з ребер графу
        edge_stats: dict[int, list[tuple[int, float]]] = {}
        for edge in optimizer.edges:
            for fid in (edge.from_id, edge.to_id):
                if 0 <= fid < num_frames:
                    edge_stats.setdefault(fid, []).append((edge.inliers, edge.rmse))

        for fid, stats in edge_stats.items():
            # РОБИМО РОЗРАХУНОК ТІЛЬКИ ДЛЯ ВАЛІДНИХ КАДРІВ
            if fid < num_frames and frame_valid[fid]:
                inliers_list = [s[0] for s in stats]
                rmse_list = [s[1] for s in stats if s[1] > 0]
                frame_matches[fid] = int(np.mean(inliers_list)) if inliers_list else 0
                frame_rmse[fid] = float(np.mean(rmse_list)) if rmse_list else 0.0

        # Disagreement: для кадрів із ≥2 ребрами, порівнюємо predictions
        # (simplified: використовуємо std відхилень у tx, ty)

        # O(E) Optical optimization

        adj = defaultdict(list)
        for e in optimizer.edges:
            adj[e.from_id].append(e)
            adj[e.to_id].append(e)

        for fid in range(num_frames):
            if not frame_valid[fid]:
                continue
            edges_to_fid = adj[fid]
            if len(edges_to_fid) >= 2:
                predictions_tx = []
                for e in edges_to_fid[:5]:  # Обмежуємо для швидкодії
                    other_id = e.from_id if e.to_id == fid else e.to_id
                    other_affine = results.get(other_id)

                    # Перевіряємо, чи сусідній кадр також валідний
                    if other_affine is not None:
                        comp = decompose_affine(other_affine)
                        predictions_tx.append(comp[0])  # tx
                if len(predictions_tx) >= 2:
                    frame_disagreement[fid] = float(np.std(predictions_tx))

        # --- Збереження в HDF5 ---
        # Тримаємо лок БД на весь цикл close → write → reload: інакше
        # конкурентний get_local_features із GUI/трекінгу впаде на закритому
        # h5py-хендлі (RuntimeError у кращому разі, сегфолт у гіршому).
        db_path = self.database.db_path
        self.database.lock.acquire()
        self.database.close()
        try:
            with h5py.File(db_path, "a") as f:
                if "calibration" in f:
                    del f["calibration"]
                grp = f.create_group("calibration")

                grp.attrs["version"] = "3.0"  # Нова версія: графова оптимізація
                grp.attrs["num_anchors"] = len(anchors)
                grp.attrs["anchors_json"] = json.dumps(
                    [a.to_dict() for a in anchors], ensure_ascii=False
                )
                grp.attrs["projection_json"] = json.dumps(
                    self.calibration.converter.export_metadata()
                )
                grp.attrs["optimizer"] = "pose_graph_lm"
                grp.attrs["num_temporal_edges"] = sum(
                    1 for e in optimizer.edges if e.edge_type == "temporal"
                )
                grp.attrs["num_spatial_edges"] = sum(
                    1 for e in optimizer.edges if e.edge_type == "spatial"
                )

                grp.create_dataset("frame_affine", data=frame_affine, compression="gzip")
                grp.create_dataset(
                    "frame_valid", data=frame_valid.astype(np.uint8), compression="gzip"
                )
                grp.create_dataset("frame_rmse", data=frame_rmse, compression="gzip")
                grp.create_dataset(
                    "frame_disagreement", data=frame_disagreement, compression="gzip"
                )
                grp.create_dataset("frame_matches", data=frame_matches, compression="gzip")

                # Обчислюємо та зберігаємо frame_gps (lat/lon для кожного кадру)
                # Для мультиджерельної геолокалізації — дозволяє SpatialIndex
                if self.calibration.converter and self.calibration.converter.is_initialized:
                    frame_gps = np.full((num_frames, 2), np.nan, dtype=np.float64)
                    gps_count = 0
                    for fid in range(num_frames):
                        if not frame_valid[fid]:
                            continue
                        affine = frame_affine[fid]
                        # Центр кадру в пікселях → metric через affine
                        center_px = np.array(
                            [[self.frame_w / 2.0, self.frame_h / 2.0]], dtype=np.float64
                        )
                        center_metric = GeometryTransforms.apply_affine(center_px, affine)
                        if center_metric is not None and len(center_metric) > 0:
                            try:
                                lat, lon = self.calibration.converter.metric_to_gps(
                                    float(center_metric[0, 0]),
                                    float(center_metric[0, 1]),
                                )
                                frame_gps[fid] = [lat, lon]
                                gps_count += 1
                            except Exception:
                                pass

                    # Видаляємо старий датасет якщо є
                    if "frame_gps" in f:
                        del f["frame_gps"]
                    f.create_dataset("frame_gps", data=frame_gps, compression="gzip")
                    logger.info(
                        f"Saved frame_gps: {gps_count}/{num_frames} frames with GPS coordinates"
                    )

            valid_count = int(np.sum(frame_valid))
            logger.success(
                f"Graph propagation saved to HDF5 (v3.0, "
                f"{len(anchors)} anchors, {optimizer.num_edges} edges, "
                f"{valid_count}/{num_frames} valid frames)"
            )
        finally:
            try:
                self.database._load_hot_data()
            finally:
                self.database.lock.release()

        return int(np.sum(frame_valid))

    # ─── Допоміжні методи ────────────────────────────────────────────────────

    def _load_previous_affines(self) -> dict[int, np.ndarray]:
        """Завантажує frame_affine попереднього калібрування з HDF5 (warm start)."""
        try:
            with h5py.File(self.database.db_path, "r") as f:
                if "calibration" not in f or "frame_affine" not in f["calibration"]:
                    return {}
                fa = f["calibration"]["frame_affine"][:]
                fv = f["calibration"]["frame_valid"][:].astype(bool)
            return {i: fa[i] for i in range(len(fa)) if fv[i]}
        except Exception as e:
            logger.warning(f"Warm start: не вдалось прочитати попередній розв'язок: {e}")
            return {}

    def _match_and_build_edge(
        self, features_a: dict, features_b: dict
    ) -> tuple[np.ndarray, int, float, int] | None:
        """Матчить дві фічі та повертає (H, inliers, rmse, n_matches) або None.

        H maps features_a (src) → features_b (dst).
        """
        try:
            mkpts_a, mkpts_b = self.matcher.match(features_a, features_b)
            if (
                len(mkpts_a) < self.min_matches
                and self.mnn_fallback
                and hasattr(self.matcher, "match_mnn")
            ):
                # Етап 8: LightGlue «сліпне» на повторюваній ріллі (12–28 матчів
                # там, де MNN по тих самих дескрипторах бачить 100–800 пар).
                mkpts_a, mkpts_b = self.matcher.match_mnn(features_a, features_b)
            if len(mkpts_a) < self.min_matches:
                return None

            H, mask = GeometryTransforms.estimate_homography(
                mkpts_a,
                mkpts_b,
                ransac_threshold=self.ransac_thresh,
                backend=self.homography_backend,
                use_mad_ransac=self.use_mad_ransac,
                mad_k_factor=self.mad_k_factor,
            )
            if H is None or mask is None:
                return None

            inlier_mask = mask.ravel().astype(bool)
            inliers = int(np.sum(inlier_mask))
            if inliers < self.min_matches:
                return None

            # RMSE
            pts_a_in = mkpts_a[inlier_mask]
            pts_transformed = GeometryTransforms.apply_homography(pts_a_in, H)
            pts_b_in = mkpts_b[inlier_mask]
            rmse = float(np.sqrt(np.mean(np.sum((pts_transformed - pts_b_in) ** 2, axis=1))))

            return H, inliers, rmse, int(len(mkpts_a))
        except Exception:
            return None

    @staticmethod
    def _compute_weight(inliers: int, rmse: float, base_weight: float) -> float:
        """Обчислює вагу ребра: w = base * √inliers / (1 + RMSE)."""
        return base_weight * np.sqrt(max(inliers, 1)) / (1.0 + rmse)

    def _fill_gaps_pchip(self, frame_affine, frame_valid, valid_ids):
        """PCHIP-заповнення (Етап 4): центр-базова shape-preserving 5-DoF інтерполяція
        над УСІМА валідними кадрами (спільний білдер із MultiAnchorCalibration).
        Кадри поза діапазоном валідних — clamp до крайнього (як лінійна екстраполяція).
        None → білдер не зібрав інтерполятор → fallback на лінійну."""
        from src.geometry.affine_utils import build_5dof_pchip, sample_5dof_pchip

        ref_px = (self.frame_w / 2.0, self.frame_h / 2.0)
        affines = [frame_affine[int(i)] for i in valid_ids]
        interp, sign, rng = build_5dof_pchip(
            valid_ids, affines, ref_px, log_scale=self.log_scale_interp
        )
        if interp is None:
            return None
        filled = 0
        for fid in range(len(frame_valid)):
            if frame_valid[fid]:
                continue
            M = sample_5dof_pchip(
                interp, sign, rng, ref_px, fid, log_scale=self.log_scale_interp
            )
            if M is None:
                continue
            frame_affine[fid] = M
            frame_valid[fid] = True
            filled += 1
        return filled

    def _fill_gaps_by_interpolation(self, frame_affine: np.ndarray, frame_valid: np.ndarray) -> int:
        """5-DoF інтерполяція кадрів, пропущених через Keyframe Selection.

        Дефолт — посегментна лінійна. За прапорцем pchip_gap_fill — shape-preserving
        PCHIP над усіма валідними кадрами (Етап 4), що прибирає «сходинки» на дугах.
        """
        valid_ids = np.where(frame_valid)[0]
        if len(valid_ids) < 1:
            return 0

        if self.pchip_gap_fill and len(valid_ids) >= 2:
            pchip_filled = self._fill_gaps_pchip(frame_affine, frame_valid, valid_ids)
            if pchip_filled is not None:
                return pchip_filled  # інакше — fallback на лінійну нижче

        filled = 0

        # Екстраполяція на початок
        first_valid = valid_ids[0]
        for mid in range(0, first_valid):
            frame_affine[mid] = frame_affine[first_valid].copy()
            frame_valid[mid] = True
            filled += 1

        # Інтерполяція розривів всередині траєкторії
        if len(valid_ids) >= 2:
            for i in range(len(valid_ids) - 1):
                left = valid_ids[i]
                right = valid_ids[i + 1]
                gap = right - left
                if gap <= 1:
                    continue

                # ВИКОРИСТОВУЄМО 5-DoF ДЕКОМПОЗИЦІЮ
                det = np.linalg.det(frame_affine[left][:2, :2])
                sign = -1.0 if det < 0 else 1.0
                comp_left = np.array(decompose_affine_5dof(frame_affine[left]), dtype=np.float64)
                comp_right = np.array(decompose_affine_5dof(frame_affine[right]), dtype=np.float64)

                # Запобігаємо стрибкам кута (кут тепер під індексом 4)
                angles = unwrap_angles([comp_left[4], comp_right[4]])
                comp_left[4] = angles[0]
                comp_right[4] = angles[1]

                # Log-scale (RESEARCH 1.3): лінійна інтерполяція в log-просторі
                # масштабу = геометрична інтерполяція самого масштабу.
                if self.log_scale_interp:
                    comp_left[2:4] = np.log(np.maximum(comp_left[2:4], 1e-12))
                    comp_right[2:4] = np.log(np.maximum(comp_right[2:4], 1e-12))

                for mid in range(left + 1, right):
                    t = (mid - left) / gap
                    comp_mid = comp_left * (1.0 - t) + comp_right * t

                    # Розпаковуємо 5 змінних
                    tx, ty, sx, sy, angle = comp_mid
                    if self.log_scale_interp:
                        sx, sy = float(np.exp(sx)), float(np.exp(sy))
                    sx = float(np.clip(sx, 1e-6, 1e6))
                    sy = float(np.clip(sy, 1e-6, 1e6))

                    # ВИКОРИСТОВУЄМО 5-DoF КОМПОЗИЦІЮ ЗІ ЗБЕРЕЖЕННЯМ ВІДОБРАЖЕННЯ
                    frame_affine[mid] = compose_affine_5dof(
                        float(tx), float(ty), sx, sy, float(angle), sign=sign
                    )
                    frame_valid[mid] = True
                    filled += 1

        # Екстраполяція на кінець
        last_valid = valid_ids[-1]
        for mid in range(last_valid + 1, len(frame_valid)):
            frame_affine[mid] = frame_affine[last_valid].copy()
            frame_valid[mid] = True
            filled += 1

        return filled
