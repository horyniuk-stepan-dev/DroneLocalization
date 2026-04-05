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
from PyQt6.QtCore import QThread, pyqtSignal

from config.config import get_cfg
from src.geometry.affine_utils import (
    compose_affine_5dof,
    decompose_affine,
    decompose_affine_5dof,
    unwrap_angles,
)
from src.geometry.pose_graph_optimizer import (
    PoseGraphOptimizer,
    homography_to_similarity,
)
from src.geometry.transformations import GeometryTransforms
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CalibrationPropagationWorker(QThread):
    """
    Графова пропагація з глобальною оптимізацією.

    Фази:
      1. Prefetch фіч → побудова часових ребер (sequential matching)
      2. Loop closure detection (FAISS DINOv2 retrieval → LightGlue matching)
      3. Фіксація GPS-якорів + BFS ініціалізація початкового наближення
      4. Глобальна оптимізація (Levenberg-Marquardt)
      5. Збереження результатів у HDF5
    """

    progress = pyqtSignal(int, str)
    completed = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, database, calibration, matcher, config=None):
        super().__init__()
        self.database = database
        self.calibration = calibration
        self.matcher = matcher
        self.config = config or {}
        self._is_running = True

        self.min_matches = get_cfg(self.config, "localization.min_matches", 15)
        self.ransac_thresh = get_cfg(self.config, "localization.ransac_threshold", 3.0)

        self.frame_w = self.database.metadata.get("frame_width", 1920)
        self.frame_h = self.database.metadata.get("frame_height", 1080)

        # Параметри графової оптимізації
        self.lc_top_k = get_cfg(self.config, "graph_optimization.loop_closure_top_k", 5)
        self.lc_min_sim = get_cfg(
            self.config, "graph_optimization.loop_closure_min_similarity", 0.75
        )
        self.lc_min_gap = get_cfg(self.config, "graph_optimization.loop_closure_min_frame_gap", 3)
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

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            self._propagate()
        except Exception as e:
            logger.error(
                f"Graph propagation failed: {e} | "
                f"num_anchors={len(self.calibration.anchors)}, "
                f"db_frames={self.database.get_num_frames()}",
                exc_info=True,
            )
            self.error.emit(str(e))

    # ─── Головний метод ──────────────────────────────────────────────────────

    def _propagate(self):
        num_frames = self.database.get_num_frames()
        all_anchors = sorted(self.calibration.anchors, key=lambda a: a.frame_id)
        anchors = [a for a in all_anchors if a.frame_id < num_frames]

        if not anchors:
            self.error.emit("Немає якорів калібрування")
            return

        logger.info(
            f"Starting GRAPH propagation for {num_frames} frames "
            f"using {len(anchors)} anchors: "
            f"{[f'#{a.frame_id}' for a in anchors]}"
        )

        # ── Phase 1: Prefetch + Temporal edges ───────────────────────────────
        self.progress.emit(0, "Передзавантаження фіч у RAM...")
        all_features = self._prefetch_features(num_frames)

        optimizer = PoseGraphOptimizer(self.frame_w, self.frame_h)
        for i in range(num_frames):
            if i in all_features:
                optimizer.add_node(i)

        self.progress.emit(10, "Побудова часових ребер (sequential matching)...")
        temporal_count = self._build_temporal_edges(optimizer, all_features, num_frames)
        logger.info(f"Phase 1 complete: {temporal_count} temporal edges")

        # ── Phase 2: Loop closure detection ──────────────────────────────────
        self.progress.emit(30, "Пошук просторових замикань (loop closure)...")
        spatial_count = self._detect_loop_closures(optimizer, all_features, num_frames)
        logger.info(f"Phase 2 complete: {spatial_count} spatial edges (loop closures)")
        logger.info(
            f"Graph: {optimizer.num_nodes} nodes, {optimizer.num_edges} edges "
            f"({temporal_count} temporal + {spatial_count} spatial)"
        )

        # ── Phase 3: Fix anchors + BFS initialization ───────────────────────
        self.progress.emit(60, "Фіксація GPS-якорів та ініціалізація графу...")
        for anchor in anchors:
            optimizer.fix_node(anchor.frame_id, anchor.affine_matrix)

        if self.use_bfs:
            bfs_count = optimizer.initialize_from_bfs()
            logger.info(f"Phase 3 complete: {bfs_count} nodes initialized via BFS")
        else:
            logger.info("Phase 3 complete: BFS initialization skipped (disabled)")

        # ── Phase 4: Optimize ────────────────────────────────────────────────
        self.progress.emit(70, "Глобальна оптимізація графу (Levenberg-Marquardt)...")
        results = optimizer.optimize(
            max_iterations=self.max_iters,
            tolerance=self.tolerance,
        )
        logger.info(f"Phase 4 complete: {len(results)} frames optimized")

        # ── Phase 5: Save to HDF5 ───────────────────────────────────────────
        self.progress.emit(85, "Збереження результатів у HDF5...")
        valid_count = self._save_to_hdf5(results, anchors, optimizer)

        # Експорт GeoJSON для візуалізації
        if self.export_geojson and self.calibration.converter:
            try:
                geojson = optimizer.export_graph_geojson(
                    self.calibration.converter, self.frame_w, self.frame_h
                )
                geojson_path = str(self.database.db_path).replace(".h5", "_graph.geojson")
                with open(geojson_path, "w", encoding="utf-8") as f:
                    json.dump(geojson, f, indent=2, ensure_ascii=False)
                logger.success(f"Graph exported to GeoJSON: {geojson_path}")
            except Exception as e:
                logger.warning(f"GeoJSON export failed: {e}")

        self.progress.emit(
            100,
            f"Готово! {valid_count}/{num_frames} кадрів отримали координати "
            f"({temporal_count} часових + {spatial_count} просторових ребер).",
        )
        self.completed.emit()

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
                self.progress.emit(
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
        last_success_id = -1
        last_success_feat = None

        for i in range(num_frames):
            if not self._is_running:
                break

            feat_i = features.get(i)
            if feat_i is None:
                continue

            if last_success_feat is not None and (i - last_success_id) <= self.max_skip_frames:
                result = self._match_and_build_edge(feat_i, last_success_feat)
                if result is not None:
                    H, inliers, rmse_val = result
                    similarity = homography_to_similarity(H, self.frame_w, self.frame_h)
                    if similarity is not None:
                        weight = self._compute_weight(inliers, rmse_val, self.temporal_base_w)
                        optimizer.add_edge(
                            from_id=last_success_id,
                            to_id=i,
                            relative_affine_2x3=similarity,
                            weight=weight,
                            edge_type="temporal",
                            inliers=inliers,
                            rmse=rmse_val,
                        )
                        count += 1

            last_success_id = i
            last_success_feat = feat_i

            if i % 200 == 0:
                self.progress.emit(
                    10 + int(i / num_frames * 18),
                    f"Часові ребра: {count} (кадр {i}/{num_frames})",
                )

        return count

    # ─── Phase 2: Loop closure detection ─────────────────────────────────────

    def _detect_loop_closures(
        self,
        optimizer: PoseGraphOptimizer,
        features: dict,
        num_frames: int,
    ) -> int:
        """Знаходить просторові замикання через DINOv2 + LightGlue matching."""
        # Побудова FAISS індексу
        global_desc = self.database.global_descriptors
        if global_desc is None or len(global_desc) == 0:
            logger.warning("No global descriptors — skipping loop closure detection")
            return 0

        dim = global_desc.shape[1]
        normed = global_desc / (np.linalg.norm(global_desc, axis=1, keepdims=True) + 1e-8)
        normed = normed.astype(np.float32)

        index = faiss.IndexFlatIP(dim)
        index.add(normed)
        logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

        count = 0
        already_matched: set[tuple[int, int]] = set()

        for i in range(num_frames):
            if not self._is_running:
                break

            feat_i = features.get(i)
            if feat_i is None:
                continue

            # Query top-K кандидатів
            q = normed[i : i + 1]
            scores, ids = index.search(q, self.lc_top_k + 1)  # +1 бо сам себе знайде

            for raw_j, sim_score in zip(ids[0], scores[0]):
                j = int(raw_j)
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

                feat_j = features.get(j)
                if feat_j is None:
                    continue

                # Matching: feat_j → feat_i (H maps j pixels → i pixels)
                result = self._match_and_build_edge(feat_j, feat_i)
                if result is None:
                    continue

                H, inliers, rmse_val = result
                if inliers < self.lc_min_inliers:
                    continue

                similarity = homography_to_similarity(H, self.frame_w, self.frame_h)
                if similarity is None:
                    continue

                weight = self._compute_weight(inliers, rmse_val, self.spatial_base_w)
                optimizer.add_edge(
                    from_id=i,
                    to_id=j,
                    relative_affine_2x3=similarity,
                    weight=weight,
                    edge_type="spatial",
                    inliers=inliers,
                    rmse=rmse_val,
                )
                count += 1

            if i % 200 == 0:
                self.progress.emit(
                    30 + int(i / num_frames * 28),
                    f"Loop closure: {count} знайдено (кадр {i}/{num_frames})",
                )

        return count

    # ─── Phase 5: Save to HDF5 ───────────────────────────────────────────────

    def _save_to_hdf5(
        self,
        results: dict[int, np.ndarray],
        anchors,
        optimizer: PoseGraphOptimizer,
    ) -> int:
        """Зберігає оптимізовані афінні матриці у HDF5.

        Формат 100% сумісний з існуючим DatabaseLoader.
        """
        num_frames = self.database.get_num_frames()
        frame_affine = np.zeros((num_frames, 2, 3), dtype=np.float32)
        frame_valid = np.zeros(num_frames, dtype=bool)
        frame_rmse = np.zeros(num_frames, dtype=np.float32)
        frame_disagreement = np.zeros(num_frames, dtype=np.float32)
        frame_matches = np.zeros(num_frames, dtype=np.int32)

        # Записуємо результати оптимізації
        # Оскільки optimizer повертає ТІЛЬКИ досяжні вузли,
        # незв'язані кадри залишаться з frame_valid = False
        for frame_id, affine in results.items():
            if 0 <= frame_id < num_frames:
                frame_affine[frame_id] = affine.astype(np.float32)
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
        db_path = self.database.db_path
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

            valid_count = int(np.sum(frame_valid))
            logger.success(
                f"Graph propagation saved to HDF5 (v3.0, "
                f"{len(anchors)} anchors, {optimizer.num_edges} edges, "
                f"{valid_count}/{num_frames} valid frames)"
            )
        finally:
            self.database._load_hot_data()

        return int(np.sum(frame_valid))

    # ─── Допоміжні методи ────────────────────────────────────────────────────

    def _match_and_build_edge(
        self, features_a: dict, features_b: dict
    ) -> tuple[np.ndarray, int, float] | None:
        """Матчить дві фічі та повертає (H, inliers, rmse) або None.

        H maps features_a (src) → features_b (dst).
        """
        try:
            mkpts_a, mkpts_b = self.matcher.match(features_a, features_b)
            if len(mkpts_a) < self.min_matches:
                return None

            H, mask = GeometryTransforms.estimate_homography(
                mkpts_a, mkpts_b, ransac_threshold=self.ransac_thresh
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

            return H, inliers, rmse
        except Exception:
            return None

    @staticmethod
    def _compute_weight(inliers: int, rmse: float, base_weight: float) -> float:
        """Обчислює вагу ребра: w = base * √inliers / (1 + RMSE)."""
        return base_weight * np.sqrt(max(inliers, 1)) / (1.0 + rmse)

    def _fill_gaps_by_interpolation(self, frame_affine: np.ndarray, frame_valid: np.ndarray) -> int:
        """Лінійна 5-DoF інтерполяція для кадрів, пропущених через Keyframe Selection."""
        valid_ids = np.where(frame_valid)[0]
        if len(valid_ids) < 1:
            return 0

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
                comp_left = np.array(decompose_affine_5dof(frame_affine[left]), dtype=np.float64)
                comp_right = np.array(decompose_affine_5dof(frame_affine[right]), dtype=np.float64)

                # Запобігаємо стрибкам кута (кут тепер під індексом 4)
                angles = unwrap_angles([comp_left[4], comp_right[4]])
                comp_left[4] = angles[0]
                comp_right[4] = angles[1]

                for mid in range(left + 1, right):
                    t = (mid - left) / gap
                    comp_mid = comp_left * (1.0 - t) + comp_right * t

                    # Розпаковуємо 5 змінних
                    tx, ty, sx, sy, angle = comp_mid
                    sx = float(np.clip(sx, 1e-6, 1e6))
                    sy = float(np.clip(sy, 1e-6, 1e6))

                    # ВИКОРИСТОВУЄМО 5-DoF КОМПОЗИЦІЮ
                    frame_affine[mid] = compose_affine_5dof(
                        float(tx), float(ty), sx, sy, float(angle)
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
