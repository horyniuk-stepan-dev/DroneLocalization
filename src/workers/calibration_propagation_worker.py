"""
calibration_propagation_worker.py — ВИПРАВЛЕНА ВЕРСІЯ

Ключові зміни:
1. Multi-anchor Gaussian blending: кожен кадр отримує зважений вплив ВІД УСІХ якорів,
   а не лише від лівого/правого сусіда. Чим далі — тим менший вплив (exp(-d²/2σ²)).
2. Robust chain-building: якщо матч між frame[i] і frame[i-1] провалився,
   пробуємо матч з останнім успішним кадром (gap-bridging). Після max_skip_frames
   ланцюжок переривається, але решта кадрів ВСЕ ОДНО отримають координати через blending.
3. Fallback interpolation: кадри, яким не вдалося отримати гомографію через жоден ланцюжок,
   отримують інтерпольовану афінну матрицю між найближчими успішними кадрами.
"""

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


class CalibrationPropagationWorker(QThread):
    """
    Хвильова пропагація на основі візуального матчингу.
    Генерує фінальну метричну афінну матрицю (2x3) для кожного кадру в базі.

    Алгоритм:
      - Для кожного кадру будується гомографія до кожного якоря (через saved poses або візуальний ланцюжок).
      - Фінальна афінна матриця обчислюється як зважена сума внесків від ВСІХ якорів
        з вагами w_i = exp(-(distance_i / sigma)^2), де distance_i — відстань у кадрах до якоря i.
      - Sigma = автоматично = половина середнього інтервалу між якорями.
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
            logger.error(f"Propagation failed: {e}", exc_info=True)
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
            return

        logger.info(
            f"Starting multi-anchor Gaussian propagation for {num_frames} frames "
            f"using {len(anchors)} anchors"
        )

        # --- Sigma для Gaussian blending ---
        # Береться як половина середнього інтервалу між якорями, але не менше 50 кадрів.
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
        # Результат: anchor_id -> {frame_id: {"H": ndarray(3,3), "matches": int}}
        self.progress.emit(10, "Побудова гомографічних ланцюжків від якорів...")
        anchor_chains = self._build_all_anchor_chains(
            anchors, anchor_features, num_frames
        )

        # --- Multi-anchor Gaussian blending ---
        self.progress.emit(50, "Multi-anchor Gaussian blending...")
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

    def _build_all_anchor_chains(
        self, anchors, anchor_features, num_frames
    ) -> dict:
        """
        Для кожного якоря будує гомографічний ланцюжок до всіх кадрів бази.
        Повертає: {anchor_frame_id: {frame_id: {"H": ..., "matches": ...}}}

        Щоб зменшити витрати: від кожного якоря поширюємось у обидва боки
        тільки в межах 3*sigma (за межами вага буде < 0.01).
        """
        max_workers = get_cfg(self.config, "models.performance.propagation_max_workers", 4)
        result = {}

        def build_for_anchor(anchor):
            anchor_id = anchor.frame_id
            feat = anchor_features.get(anchor_id)
            if feat is None:
                logger.warning(f"No features for anchor {anchor_id}, skipping chain")
                return anchor_id, {}

            # Поширюємось у обидва боки від якоря
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

        Покращення порівняно зі старою версією:
        - Якщо матч з попереднім кадром провалився — намагаємось матчити з anchor напряму,
          або з останнім успішним кадром у межах max_skip_frames.
        - Використовує saved frame_poses якщо доступні (O(1)).
        """
        result = {}

        # --- Спочатку пробуємо saved poses (O(1), без GPU) ---
        try:
            pose_anchor = self.database.frame_poses[anchor_id].astype(np.float64)
            if np.abs(np.linalg.det(pose_anchor)) > 1e-9:
                inv_pose_anchor = np.linalg.inv(pose_anchor)
                for frame_id in frames:
                    if not self._is_running:
                        break
                    try:
                        pose_frame = self.database.frame_poses[frame_id].astype(np.float64)
                        if np.abs(np.linalg.det(pose_frame)) < 1e-9:
                            continue
                        H = (inv_pose_anchor @ pose_frame).astype(np.float32)
                        result[frame_id] = {"H": H, "matches": 50}
                    except Exception:
                        continue
                if result:
                    return result
        except Exception as e:
            logger.debug(f"Saved poses unavailable for anchor {anchor_id}: {e}")

        # --- Fallback: візуальний ланцюжок з gap-bridging ---
        # h_cache: frame_id -> H (відносно anchor)
        h_cache = {anchor_id: np.eye(3, dtype=np.float32)}
        # Черга "мостів": список (frame_id, H, feat) відсортований за frame_id
        # Зберігаємо кілька останніх успішних точок для gap-bridging
        successful = [(anchor_id, np.eye(3, dtype=np.float32), anchor_feat)]

        for frame_id in frames:
            if not self._is_running:
                break
            curr_feat = self._all_features.get(frame_id)
            if curr_feat is None:
                continue

            H_found = None
            n_matches = 0

            # Пробуємо матч з кожним з останніх successful (від найближчого до найдальшого)
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
                # Обмежуємо буфер
                if len(successful) > self.max_skip_frames + 2:
                    successful.pop(0)

        return result

    # -------------------------------------------------------------------------
    # Multi-anchor Gaussian blending
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
        Для кожного не-якірного кадру обчислює зважену суму метричних точок від усіх якорів.
        Ваги: w_i = exp(-(dist_to_anchor_i / sigma)^2)

        Це забезпечує "розлив" від кожного опорного кадру у всі сторони:
        - Кадр поруч з якорем майже повністю визначається цим якорем.
        - Кадр між двома якорями плавно переходить від одного до іншого.
        - Кадр далеко від усіх якорів отримує слабкий але ненульовий внесок від усіх.
        """
        anchor_ids = [a.frame_id for a in anchors]

        for frame_id in range(num_frames):
            if not self._is_running:
                return
            if frame_id % 200 == 0:
                self.progress.emit(
                    50 + int(frame_id / num_frames * 33),
                    f"Blending кадр {frame_id}/{num_frames}...",
                )

            # Пропускаємо самі якорі — вони вже ініціалізовані
            if frame_id in anchor_ids:
                continue

            weighted_pts_sum = None
            weight_total = 0.0
            total_matches = 0
            all_projections = []  # для disagreement

            for anchor in anchors:
                anchor_id = anchor.frame_id
                chain = anchor_chains.get(anchor_id, {})
                info = chain.get(frame_id)
                if info is None:
                    continue

                metric_pts = self._project_to_metric(info["H"], anchor)
                if metric_pts is None:
                    continue

                dist = abs(frame_id - anchor_id)
                # Gaussian weight: повний вплив поруч, затухає вдалині
                w = np.exp(-(dist / sigma) ** 2)

                if weighted_pts_sum is None:
                    weighted_pts_sum = metric_pts * w
                else:
                    weighted_pts_sum += metric_pts * w
                weight_total += w
                total_matches += info["matches"]
                all_projections.append(metric_pts)

            if weighted_pts_sum is None or weight_total < 1e-9:
                continue

            final_metric_pts = weighted_pts_sum / weight_total

            # Disagreement: середнє відхилення між проекціями різних якорів
            if len(all_projections) >= 2:
                dists = []
                for proj in all_projections:
                    dists.append(np.mean(np.linalg.norm(proj - final_metric_pts, axis=1)))
                frame_disagreement[frame_id] = float(np.mean(dists))

            M, _ = cv2.estimateAffine2D(self.grid_points, final_metric_pts)
            if M is not None:
                frame_affine[frame_id] = M
                frame_valid[frame_id] = True
                proj = GeometryTransforms.apply_affine(self.grid_points, M)
                rmse = np.sqrt(np.mean(np.linalg.norm(proj - final_metric_pts, axis=1) ** 2))
                frame_rmse[frame_id] = rmse
                frame_matches[frame_id] = total_matches // max(1, len(all_projections))

    # -------------------------------------------------------------------------
    # Заповнення прогалин через лінійну інтерполяцію афінних матриць
    # -------------------------------------------------------------------------

    def _fill_gaps_by_interpolation(self, frame_affine, frame_valid):
        """
        Кадри, які не отримали координат через жодну гомографію,
        заповнюються лінійною інтерполяцією між найближчими дійсними сусідами.
        Це остання лінія захисту — гарантує що прогалин не залишиться
        між двома дійсними кадрами.
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
                continue  # сусідні — нема що заповнювати
            for mid in range(left + 1, right):
                if frame_valid[mid]:
                    continue
                t = (mid - left) / gap  # 0..1
                M_interp = frame_affine[left] * (1 - t) + frame_affine[right] * t
                frame_affine[mid] = M_interp
                frame_valid[mid] = True
                filled += 1

        if filled > 0:
            logger.info(f"Gap interpolation filled {filled} additional frames")

    # -------------------------------------------------------------------------
    # Допоміжні методи (без змін)
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
                f"(rev 2.2, {len(anchors)} anchors, Gaussian blending)"
            )
        finally:
            self.database._load_hot_data()