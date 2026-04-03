
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import h5py
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config.config import get_cfg
from src.geometry.transformations import GeometryTransforms
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Утиліти декомпозиції/складання афінних матриць — єдине джерело: affine_utils
# ---------------------------------------------------------------------------
from src.geometry.affine_utils import (
    compose_affine as _compose_affine,
    decompose_affine as _decompose_affine,
)


class CalibrationPropagationWorker(QThread):
    """
    Хвильова пропагація на основі візуального матчингу.
    Генерує фінальну метричну афінну матрицю (2x3) для кожного кадру в базі.

    Алгоритм:
      - Для кожного кадру будується гомографія до кожного якоря (через saved poses або візуальний ланцюжок).
      - Фінальна афінна матриця обчислюється через сегментний розподіл похибки замикання
        (loop closure error distribution) між парами сусідніх GPS-якорів.
      - Кути дрейфу коригуються через справжню delta між передбаченим та реальним кутом правого якоря.
      - Кадри без жодної успішної гомографії отримують інтерполяцію між найближчими сусідами.
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

        # Скільки кадрів можна "перестрибнути" при побудові ланцюжка,
        # якщо матч з безпосереднім сусідом провалився.
        self.max_skip_frames = get_cfg(self.config, "propagation.max_skip_frames", 5)

        # Базова сітка точок для точної апроксимації афінної матриці (4x4)
        grid_x = np.linspace(0, self.frame_w, 4)
        grid_y = np.linspace(0, self.frame_h, 4)
        gx, gy = np.meshgrid(grid_x, grid_y)
        self.grid_points = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            self._propagate()
        except Exception as e:
            logger.error(
                f"Propagation failed: {e} | "
                f"num_anchors={len(self.calibration.anchors)}, "
                f"db_frames={self.database.get_num_frames()}",
                exc_info=True,
            )
            self.error.emit(str(e))

    # -------------------------------------------------------------------------
    # Головний метод
    # -------------------------------------------------------------------------

    def _propagate(self):
        num_frames = self.database.get_num_frames()
        all_anchors = sorted(self.calibration.anchors, key=lambda a: a.frame_id)
        anchors = [a for a in all_anchors if a.frame_id < num_frames]

        if len(anchors) < len(all_anchors):
            logger.warning(
                f"Filtered {len(all_anchors) - len(anchors)} out-of-bounds anchors "
                f"(DB has {num_frames} frames)."
            )

        if not anchors:
            self.error.emit("Немає якорів калібрування")
            logger.error(
                "Propagation aborted: no valid anchors. "
                "Add at least one GPS calibration anchor before running propagation."
            )
            return

        logger.info(
            f"Starting multi-anchor loop-closure propagation for {num_frames} frames "
            f"using {len(anchors)} anchors: "
            f"{[f'#{a.frame_id}' for a in anchors]}"
        )

        # --- Sigma для Gaussian blending ---
        if len(anchors) >= 2:
            intervals = [anchors[i + 1].frame_id - anchors[i].frame_id for i in range(len(anchors) - 1)]
            sigma = max(50.0, float(np.mean(intervals)) * 0.5)
        else:
            sigma = max(50.0, num_frames * 0.2)
        logger.info(f"Gaussian sigma = {sigma:.1f} frames")

        frame_affine = np.zeros((num_frames, 2, 3), dtype=np.float32)
        frame_valid = np.zeros(num_frames, dtype=bool)
        frame_rmse = np.zeros(num_frames, dtype=np.float32)
        frame_disagreement = np.zeros(num_frames, dtype=np.float32)
        frame_matches = np.zeros(num_frames, dtype=np.int32)

        # --- Prefetch фіч ---
        self.progress.emit(0, "Передзавантаження фіч у RAM...")
        self._all_features = {}
        for i in range(num_frames):
            if not self._is_running:
                return
            try:
                self._all_features[i] = self.database.get_local_features(i)
            except Exception:
                pass
            if i % 1000 == 0:
                self.progress.emit(int(i / num_frames * 10), f"Prefetch: {i}/{num_frames}")
        logger.info(f"Prefetched features for {len(self._all_features)} frames")

        # --- Ініціалізація якорів ---
        anchor_features = {}
        for anchor in anchors:
            feat = self._all_features.get(anchor.frame_id)
            if feat is not None:
                anchor_features[anchor.frame_id] = feat
            frame_affine[anchor.frame_id] = anchor.affine_matrix
            frame_valid[anchor.frame_id] = True
            frame_rmse[anchor.frame_id] = getattr(anchor, "rmse_m", 0.0)
            frame_matches[anchor.frame_id] = getattr(anchor, "inliers_count", 0)

        # --- Будуємо гомографічні ланцюжки від кожного якоря до всіх кадрів ---
        self.progress.emit(10, "Побудова гомографічних ланцюжків від якорів...")
        anchor_chains = self._build_all_anchor_chains(anchors, anchor_features, num_frames)

        # --- Loop-closure blending ---
        self.progress.emit(50, "Loop-closure blending...")
        self._blend_all_frames(
            anchors=anchors,
            anchor_chains=anchor_chains,
            num_frames=num_frames,
            sigma=sigma,
            frame_affine=frame_affine,
            frame_valid=frame_valid,
            frame_rmse=frame_rmse,
            frame_disagreement=frame_disagreement,
            frame_matches=frame_matches,
        )

        # --- Fallback: заповнення кадрів без координат через інтерполяцію ---
        self.progress.emit(85, "Заповнення прогалин через інтерполяцію...")
        self._fill_gaps_by_interpolation(frame_affine, frame_valid)

        valid_count = int(np.sum(frame_valid))
        self.progress.emit(90, "Збереження результатів у HDF5...")
        self._save_to_hdf5(
            frame_affine, frame_valid, frame_rmse, frame_disagreement, frame_matches, anchors
        )

        self.progress.emit(100, f"Готово! {valid_count}/{num_frames} кадрів отримали координати.")
        self.completed.emit()

    # -------------------------------------------------------------------------
    # Побудова ланцюжків від кожного якоря
    # -------------------------------------------------------------------------

    def _build_all_anchor_chains(self, anchors, anchor_features, num_frames) -> dict:
        """
        Для кожного якоря будує гомографічний ланцюжок до всіх кадрів бази.
        Повертає: {anchor_frame_id: {frame_id: {"H": ndarray(3,3), "matches": int}}}
        """
        max_workers = get_cfg(self.config, "models.performance.propagation_max_workers", 4)
        result = {}

        def build_for_anchor(anchor):
            anchor_id = anchor.frame_id
            feat = anchor_features.get(anchor_id)
            if feat is None:
                logger.warning(
                    f"No features for anchor #{anchor_id} — skipping chain. "
                    f"This anchor will not contribute to propagation. "
                    f"Cause: frame may have zero keypoints in database."
                )
                return anchor_id, {}

            frames_left = list(range(anchor_id - 1, -1, -1))
            frames_right = list(range(anchor_id + 1, num_frames))

            chain_left = self._build_robust_chain(frames_left, anchor_id, feat)
            chain_right = self._build_robust_chain(frames_right, anchor_id, feat)

            chain = {anchor_id: {"H": np.eye(3, dtype=np.float32), "matches": 100}}
            chain.update(chain_left)
            chain.update(chain_right)
            return anchor_id, chain

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(build_for_anchor, a): a for a in anchors}
            total = len(futures)
            for done_count, future in enumerate(as_completed(futures)):
                if not self._is_running:
                    break
                anchor_id, chain = future.result()
                result[anchor_id] = chain
                self.progress.emit(
                    10 + int(done_count / total * 35),
                    f"Ланцюжок від якоря {anchor_id}: {len(chain)} кадрів",
                )

        return result

    def _build_robust_chain(self, frames, anchor_id, anchor_feat) -> dict:
        """
        Будує гомографічний ланцюжок від anchor_id до кожного кадру у списку `frames`.

        - Якщо матч з попереднім кадром провалився — намагаємось матчити з anchor напряму,
          або з останнім успішним кадром у межах max_skip_frames.
        """
        result = {}

        h_cache = {anchor_id: np.eye(3, dtype=np.float32)}
        successful = [(anchor_id, np.eye(3, dtype=np.float32), anchor_feat)]

        for frame_id in frames:
            if not self._is_running:
                break
            curr_feat = self._all_features.get(frame_id)
            if curr_feat is None:
                continue

            H_found = None
            n_matches = 0

            for prev_id, H_prev_to_anchor, prev_feat in reversed(successful[-self.max_skip_frames:]):
                res = self._match_pair_with_count(curr_feat, prev_feat)
                if res is not None:
                    H_curr_to_prev, n = res
                    H_found = (
                        H_prev_to_anchor.astype(np.float64) @ H_curr_to_prev.astype(np.float64)
                    ).astype(np.float32)
                    n_matches = n
                    break

            if H_found is not None:
                result[frame_id] = {"H": H_found, "matches": n_matches}
                successful.append((frame_id, H_found, curr_feat))
                if len(successful) > self.max_skip_frames + 2:
                    successful.pop(0)

        return result

    # -------------------------------------------------------------------------
    # Loop-closure blending між парами якорів
    # -------------------------------------------------------------------------

    def _blend_all_frames(
        self,
        anchors,
        anchor_chains,
        num_frames,
        sigma,
        frame_affine,
        frame_valid,
        frame_rmse,
        frame_disagreement,
        frame_matches,
    ):
        """
        Обчислює афінні матриці для кожного кадру з рівномірним розподілом
        похибки замикання (loop closure error distribution) між парами якорів.

        Алгоритм (сегментний loop closure):
          1. Для кожного сегменту між сусідніми якорями [A_left .. A_right]:
             a. Будуємо одометричний ланцюжок від A_left до кожного кадру сегменту.
             b. На фінальному якорі (A_right) обчислюємо дельту похибки:
                delta = affine_right_true − affine_predicted_by_chain_from_left.
                ВАЖЛИВО: predicted_at_right зберігає СПРАВЖНІЙ передбачений кут
                від одометрії (не перезаписується значенням правого якоря).
             c. Розподіляємо delta лінійно вздовж сегменту (t=0 на A_left → t=1 на A_right).
          2. Кадри за межами першого/останнього якоря отримують матрицю
             з найближчого якоря без корекції (екстраполяція).
        """
        anchor_ids = [a.frame_id for a in anchors]

        # ── Сегменти між парами сусідніх якорів ────────────────────────────────
        for seg_idx in range(len(anchors) - 1):
            a_left = anchors[seg_idx]
            a_right = anchors[seg_idx + 1]
            id_left = a_left.frame_id
            id_right = a_right.frame_id

            chain_left = anchor_chains.get(id_left, {})
            chain_right = anchor_chains.get(id_right, {})

            # Декомпозиція матриць якорів
            comp_left = np.array(_decompose_affine(a_left.affine_matrix), dtype=np.float64)
            comp_right = np.array(_decompose_affine(a_right.affine_matrix), dtype=np.float64)

            # Обчислюємо "передбачену" матрицю на правому якорі через одометрію від лівого
            predicted_at_right = None
            info_right_from_left = chain_left.get(id_right)
            if info_right_from_left is not None:
                metric_pts = self._project_to_metric(info_right_from_left["H"], a_left)
                if metric_pts is not None:
                    M_pred, _ = cv2.estimateAffine2D(
                        self.grid_points, metric_pts, method=cv2.LMEDS
                    )
                    if M_pred is not None:
                        predicted_at_right = np.array(
                            _decompose_affine(M_pred), dtype=np.float64
                        )

            seg_len = id_right - id_left

            for frame_id in range(id_left + 1, id_right):
                if not self._is_running:
                    return
                if frame_id in anchor_ids:
                    continue

                if frame_id % 200 == 0:
                    self.progress.emit(
                        50 + int(frame_id / num_frames * 33),
                        f"Loop-closure blending кадр {frame_id}/{num_frames}...",
                    )

                t = (frame_id - id_left) / seg_len  # 0..1

                # --- Одометрична оцінка від лівого якоря ---
                info_l = chain_left.get(frame_id)
                M_from_left = None
                if info_l is not None:
                    metric_pts = self._project_to_metric(info_l["H"], a_left)
                    if metric_pts is not None:
                        M_est, _ = cv2.estimateAffine2D(
                            self.grid_points, metric_pts, method=cv2.LMEDS
                        )
                        if M_est is not None:
                            M_from_left = M_est

                # --- Одометрична оцінка від правого якоря ---
                info_r = chain_right.get(frame_id)
                M_from_right = None
                if info_r is not None:
                    metric_pts = self._project_to_metric(info_r["H"], a_right)
                    if metric_pts is not None:
                        M_est, _ = cv2.estimateAffine2D(
                            self.grid_points, metric_pts, method=cv2.LMEDS
                        )
                        if M_est is not None:
                            M_from_right = M_est

                if M_from_left is None and M_from_right is None:
                    continue

                # --- Loop-closure корекція ---
                if M_from_left is not None and predicted_at_right is not None:
                    comp_est = np.array(_decompose_affine(M_from_left), dtype=np.float64)

                    # Розгортаємо кути ТРЬОХ точок разом:
                    # [лівий якір, поточна одометрія, правий якір]
                    # Це забезпечує коректний unwrap для всіх трьох значень одночасно.
                    angles = np.unwrap([comp_left[3], comp_est[3], comp_right[3]])
                    comp_left_uw = comp_left.copy()
                    comp_left_uw[3] = angles[0]
                    comp_est_uw = comp_est.copy()
                    comp_est_uw[3] = angles[1]
                    comp_right_uw = comp_right.copy()
                    comp_right_uw[3] = angles[2]

                    # --- ВИПРАВЛЕННЯ БАГ 1 ---
                    # Розгортаємо кут передбаченої матриці (predicted_at_right) разом із
                    # лівим якорем, щоб отримати коректне unwrap-значення для pred_uw.
                    # ВИДАЛЕНО: pred_uw[3] = angles[2]
                    # Той рядок примусово прирівнював кут pred_uw до кута правого якоря,
                    # через що delta[3] = comp_right_uw[3] - pred_uw[3] завжди = 0.
                    angles_pred = np.unwrap([comp_left[3], predicted_at_right[3]])
                    pred_uw = predicted_at_right.copy()
                    pred_uw[3] = angles_pred[1]  # справжній передбачений кут від одометрії

                    # Дельта похибки замикання: різниця між ІСТИННИМ правим якорем
                    # та тим, що одометрія ПЕРЕДБАЧИЛА для цього місця
                    delta = comp_right_uw - pred_uw

                    # Розподіляємо delta лінійно: t=0 (лівий якір) → корекція=0, t=1 → корекція=delta
                    comp_corrected = comp_est_uw + delta * t

                    scale = float(np.clip(comp_corrected[2], 1e-6, 1e6))
                    M_final = _compose_affine(
                        float(comp_corrected[0]),
                        float(comp_corrected[1]),
                        scale,
                        float(comp_corrected[3]),
                    )
                elif M_from_left is not None:
                    # Немає loop closure: використовуємо лише одометрію від лівого якоря
                    M_final = M_from_left.astype(np.float32)
                else:
                    # Лише одометрія від правого якоря (зворотна)
                    M_final = M_from_right.astype(np.float32)

                # --- Disagreement між двома якорями ---
                if M_from_left is not None and M_from_right is not None:
                    pts_l = GeometryTransforms.apply_affine(self.grid_points, M_from_left)
                    pts_r = GeometryTransforms.apply_affine(self.grid_points, M_from_right)
                    frame_disagreement[frame_id] = float(
                        np.mean(np.linalg.norm(pts_l - pts_r, axis=1))
                    )

                frame_affine[frame_id] = M_final
                frame_valid[frame_id] = True

                # RMSE між фінальною матрицею та grid_points (внутрішня якість апроксимації)
                proj = GeometryTransforms.apply_affine(self.grid_points, M_final)
                ref_pts = self._project_to_metric(
                    chain_left.get(frame_id, {}).get("H", np.eye(3)), a_left
                )
                if ref_pts is not None:
                    rmse = float(np.sqrt(np.mean(np.linalg.norm(proj - ref_pts, axis=1) ** 2)))
                    frame_rmse[frame_id] = rmse

                n_l = chain_left.get(frame_id, {}).get("matches", 0)
                n_r = chain_right.get(frame_id, {}).get("matches", 0)
                frame_matches[frame_id] = (n_l + n_r) // max(1, int(n_l > 0) + int(n_r > 0))

        # ── Кадри до першого якоря (екстраполяція ліворуч) ─────────────────────
        first_anchor = anchors[0]
        chain_first = anchor_chains.get(first_anchor.frame_id, {})
        for frame_id in range(0, first_anchor.frame_id):
            if not self._is_running:
                return
            if frame_valid[frame_id]:
                continue
            info = chain_first.get(frame_id)
            if info is None:
                continue
            metric_pts = self._project_to_metric(info["H"], first_anchor)
            if metric_pts is None:
                continue
            M, _ = cv2.estimateAffine2D(self.grid_points, metric_pts, method=cv2.LMEDS)
            if M is not None:
                frame_affine[frame_id] = M
                frame_valid[frame_id] = True
                frame_matches[frame_id] = info["matches"]

        # ── Кадри після останнього якоря (екстраполяція праворуч) ──────────────
        last_anchor = anchors[-1]
        chain_last = anchor_chains.get(last_anchor.frame_id, {})
        for frame_id in range(last_anchor.frame_id + 1, num_frames):
            if not self._is_running:
                return
            if frame_valid[frame_id]:
                continue
            info = chain_last.get(frame_id)
            if info is None:
                continue
            metric_pts = self._project_to_metric(info["H"], last_anchor)
            if metric_pts is None:
                continue
            M, _ = cv2.estimateAffine2D(self.grid_points, metric_pts, method=cv2.LMEDS)
            if M is not None:
                frame_affine[frame_id] = M
                frame_valid[frame_id] = True
                frame_matches[frame_id] = info["matches"]

    # -------------------------------------------------------------------------
    # Заповнення прогалин через лінійну інтерполяцію афінних матриць
    # -------------------------------------------------------------------------

    def _fill_gaps_by_interpolation(self, frame_affine, frame_valid):
        """
        Кадри, які не отримали координат через жодну гомографію,
        заповнюються інтерполяцією між найближчими дійсними сусідами.

        Використовується декомпозиція на (tx, ty, scale, angle) з подальшою лінійною
        інтерполяцією кожного скалярного каналу і відновленням матриці.
        """
        num_frames = len(frame_valid)
        valid_ids = np.where(frame_valid)[0]
        if len(valid_ids) < 2:
            return

        filled = 0
        for i in range(len(valid_ids) - 1):
            left = valid_ids[i]
            right = valid_ids[i + 1]
            gap = right - left
            if gap <= 1:
                continue

            comp_left = np.array(_decompose_affine(frame_affine[left]), dtype=np.float64)
            comp_right = np.array(_decompose_affine(frame_affine[right]), dtype=np.float64)

            angles = np.array([comp_left[3], comp_right[3]])
            angles = np.unwrap(angles)
            comp_left[3] = angles[0]
            comp_right[3] = angles[1]

            for mid in range(left + 1, right):
                if frame_valid[mid]:
                    continue
                t = (mid - left) / gap
                comp_mid = comp_left * (1.0 - t) + comp_right * t
                tx, ty, scale, angle = comp_mid
                scale = float(np.clip(scale, 1e-6, 1e6))
                frame_affine[mid] = _compose_affine(float(tx), float(ty), scale, float(angle))
                frame_valid[mid] = True
                filled += 1

        if filled > 0:
            logger.info(f"Gap interpolation filled {filled} additional frames")

    # -------------------------------------------------------------------------
    # Допоміжні методи
    # -------------------------------------------------------------------------

    def _match_pair_with_count(self, features_a: dict, features_b: dict) -> tuple | None:
        try:
            mkpts_a, mkpts_b = self.matcher.match(features_a, features_b)
            if len(mkpts_a) < self.min_matches:
                return None
            H, mask = GeometryTransforms.estimate_homography(
                mkpts_a, mkpts_b, ransac_threshold=self.ransac_thresh
            )
            if H is None:
                return None
            inliers = int(np.sum(mask))
            if inliers < self.min_matches:
                return None
            return H, inliers
        except Exception:
            logger.debug(
                f"Match failed between frame pair | features_a_kpts={len(features_a.get('keypoints', []))}, "
                f"features_b_kpts={len(features_b.get('keypoints', []))}"
            )
            return None

    def _project_to_metric(self, H_to_anchor, anchor):
        pts_in_anchor = GeometryTransforms.apply_homography(self.grid_points, H_to_anchor)
        if pts_in_anchor is None or len(pts_in_anchor) != len(self.grid_points):
            return None
        metric_pts = GeometryTransforms.apply_affine(pts_in_anchor, anchor.affine_matrix)
        return metric_pts

    def _save_to_hdf5(
        self, frame_affine, frame_valid, frame_rmse, frame_disagreement, frame_matches, anchors
    ):
        db_path = self.database.db_path
        self.database.close()
        try:
            with h5py.File(db_path, "a") as f:
                if "calibration" in f:
                    del f["calibration"]
                grp = f.create_group("calibration")

                grp.attrs["version"] = "2.2"
                grp.attrs["num_anchors"] = len(anchors)
                grp.attrs["anchors_json"] = json.dumps(
                    [a.to_dict() for a in anchors], ensure_ascii=False
                )
                grp.attrs["projection_json"] = json.dumps(
                    self.calibration.converter.export_metadata()
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

            logger.success(
                f"Successful propagation saved to HDF5 "
                f"(rev 2.2, {len(anchors)} anchors, loop-closure blending)"
            )
        finally:
            self.database._load_hot_data()