import json

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

    def _propagate(self):
        num_frames = self.database.get_num_frames()
        all_anchors = sorted(self.calibration.anchors, key=lambda a: a.frame_id)
        anchors = [a for a in all_anchors if a.frame_id < num_frames]
        
        if len(anchors) < len(all_anchors):
            logger.warning(f"Filtered {len(all_anchors) - len(anchors)} out-of-bounds anchors (DB has {num_frames} frames).")
            
        if not anchors:
            self.error.emit("Немає якорів калібрування")
            return

        logger.info(
            f"Starting visual wave propagation for {num_frames} frames using {len(anchors)} anchors"
        )

        frame_affine = np.zeros((num_frames, 2, 3), dtype=np.float32)
        frame_valid = np.zeros(num_frames, dtype=bool)

        # QA metrics
        frame_rmse = np.zeros(num_frames, dtype=np.float32)
        frame_disagreement = np.zeros(num_frames, dtype=np.float32)
        frame_matches = np.zeros(num_frames, dtype=np.int32)

        # Оптимізація A: Batch prefetch всіх фіч у RAM
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

        anchor_features = {}
        for anchor in anchors:
            try:
                # Вкрай важливо: використовуємо .get(), бо якір може бути на кадрі,
                # який не є keyframe і не має локальних ознак.
                feat = self._all_features.get(anchor.frame_id)
                if feat is not None:
                    anchor_features[anchor.frame_id] = feat

                frame_affine[anchor.frame_id] = anchor.affine_matrix
                frame_valid[anchor.frame_id] = True
                frame_rmse[anchor.frame_id] = getattr(anchor, "rmse_m", 0.0)
                frame_matches[anchor.frame_id] = getattr(anchor, "inliers_count", 0)
            except Exception as e:
                self.error.emit(f"Не вдалося ініціалізувати якір {anchor.frame_id}: {e}")
                return

        segments = self._build_segments(anchors, num_frames)
        total_segments = len(segments)

        # Fix 7: Паралельна propagation по незалежних сегментах
        from concurrent.futures import ThreadPoolExecutor, as_completed

        between_segments = [s for s in segments if s["type"] == "between"]
        tail_segments = [s for s in segments if s["type"] == "tail"]

        logger.info(f"Parallel processing of {len(between_segments)} segments...")
        max_workers = get_cfg(self.config, "models.performance.propagation_max_workers", 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_segment,
                    seg,
                    anchor_features,
                    frame_affine,
                    frame_valid,
                    frame_rmse,
                    frame_disagreement,
                    frame_matches,
                    i,
                    total_segments,
                    num_frames,
                ): seg
                for i, seg in enumerate(between_segments)
            }
            for future in as_completed(futures):
                if not self._is_running:
                    break
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Segment processing failed: {e}")

        # Хвости зазвичай дешевші і можуть бути послідовними або теж паралельними
        for i, seg in enumerate(tail_segments):
            if not self._is_running:
                break
            self._process_segment(
                seg,
                anchor_features,
                frame_affine,
                frame_valid,
                frame_rmse,
                frame_disagreement,
                frame_matches,
                len(between_segments) + i,
                total_segments,
                num_frames,
            )

        valid_count = int(np.sum(frame_valid))
        self.progress.emit(90, "Збереження результатів у HDF5...")
        self._save_to_hdf5(
            frame_affine, frame_valid, frame_rmse, frame_disagreement, frame_matches, anchors
        )

        self.progress.emit(100, f"Готово! {valid_count}/{num_frames} кадрів отримали координати.")
        self.completed.emit()

    def _build_segments(self, anchors: list, num_frames: int) -> list:
        segments = []
        if anchors[0].frame_id > 0:
            segments.append(
                {
                    "type": "tail",
                    "frames": list(range(anchors[0].frame_id - 1, -1, -1)),
                    "anchor": anchors[0],
                }
            )

        for i in range(len(anchors) - 1):
            left_anchor = anchors[i]
            right_anchor = anchors[i + 1]
            segments.append(
                {
                    "type": "between",
                    "left_anchor": left_anchor,
                    "right_anchor": right_anchor,
                    "frames": list(range(left_anchor.frame_id + 1, right_anchor.frame_id)),
                }
            )

        if anchors[-1].frame_id < num_frames - 1:
            segments.append(
                {
                    "type": "tail",
                    "frames": list(range(anchors[-1].frame_id + 1, num_frames)),
                    "anchor": anchors[-1],
                }
            )

        # Валідація: сегменти НЕ повинні перетинатися — це гарантує
        # thread-safety при паралельному записі в numpy-масиви через ThreadPoolExecutor
        all_frame_ids = set()
        for seg in segments:
            seg_frames = set(seg["frames"])
            overlap = all_frame_ids & seg_frames
            assert not overlap, (
                f"CRITICAL: Segment overlap detected on frames {overlap}! "
                f"This would cause race conditions in ThreadPoolExecutor."
            )
            all_frame_ids |= seg_frames

        return segments

    def _process_segment(
        self,
        segment,
        anchor_features,
        frame_affine,
        frame_valid,
        frame_rmse,
        frame_disagreement,
        frame_matches,
        seg_idx,
        total_segments,
        num_frames,
    ):
        if segment["type"] == "tail":
            anchor = segment["anchor"]
            frames = segment["frames"]
            self._wave_from_anchor(
                frames=frames,
                anchor=anchor,
                anchor_feat=anchor_features[anchor.frame_id],
                frame_affine=frame_affine,
                frame_valid=frame_valid,
                frame_rmse=frame_rmse,
                frame_matches=frame_matches,
                seg_idx=seg_idx,
                total_segments=total_segments,
                num_frames=num_frames,
            )
        elif segment["type"] == "between":
            left_anchor = segment["left_anchor"]
            right_anchor = segment["right_anchor"]
            frames = segment["frames"]

            h_left_res = self._build_homography_chain(
                frames, left_anchor, anchor_features[left_anchor.frame_id]
            )
            frames_reversed = list(reversed(frames))
            h_right_res = self._build_homography_chain(
                frames_reversed, right_anchor, anchor_features[right_anchor.frame_id]
            )

            total_frames_in_seg = len(frames)
            for local_idx, frame_id in enumerate(frames):
                if not self._is_running:
                    return

                if local_idx % 10 == 0:
                    prog = int(
                        (
                            seg_idx / total_segments
                            + local_idx / (total_frames_in_seg * total_segments)
                        )
                        * 90
                    )
                    self.progress.emit(prog, f"Блендінг: кадр {frame_id}/{num_frames}...")

                H_to_left_info = h_left_res.get(frame_id)
                H_to_right_info = h_right_res.get(frame_id)

                metric_pts_left = None
                n_left = 0
                if H_to_left_info:
                    metric_pts_left = self._project_to_metric(H_to_left_info["H"], left_anchor)
                    n_left = H_to_left_info["matches"]

                metric_pts_right = None
                n_right = 0
                if H_to_right_info:
                    metric_pts_right = self._project_to_metric(H_to_right_info["H"], right_anchor)
                    n_right = H_to_right_info["matches"]

                final_metric_pts = None
                disagreement = 0.0

                if metric_pts_left is not None and metric_pts_right is not None:
                    disagreement = np.mean(
                        np.linalg.norm(metric_pts_left - metric_pts_right, axis=1)
                    )

                    dist_to_left = abs(frame_id - left_anchor.frame_id)
                    dist_to_right = abs(frame_id - right_anchor.frame_id)
                    weight_left = dist_to_right / (dist_to_left + dist_to_right)
                    weight_right = 1.0 - weight_left
                    final_metric_pts = (
                        metric_pts_left * weight_left + metric_pts_right * weight_right
                    )
                    frame_matches[frame_id] = int((n_left + n_right) / 2)
                elif metric_pts_left is not None:
                    final_metric_pts = metric_pts_left
                    frame_matches[frame_id] = n_left
                elif metric_pts_right is not None:
                    final_metric_pts = metric_pts_right
                    frame_matches[frame_id] = n_right

                if final_metric_pts is not None:
                    M, _ = cv2.estimateAffine2D(self.grid_points, final_metric_pts)
                    if M is not None:
                        frame_affine[frame_id] = M
                        frame_valid[frame_id] = True
                        proj = GeometryTransforms.apply_affine(self.grid_points, M)
                        rmse = np.sqrt(
                            np.mean(np.linalg.norm(proj - final_metric_pts, axis=1) ** 2)
                        )
                        frame_rmse[frame_id] = rmse
                        frame_disagreement[frame_id] = disagreement

    def _wave_from_anchor(
        self,
        frames,
        anchor,
        anchor_feat,
        frame_affine,
        frame_valid,
        frame_rmse,
        frame_matches,
        seg_idx,
        total_segments,
        num_frames,
    ):
        h_chain = self._build_homography_chain(frames, anchor, anchor_feat)
        total_frames_in_seg = len(frames)

        for local_idx, frame_id in enumerate(frames):
            if not self._is_running:
                return

            if local_idx % 20 == 0:
                prog = int(
                    (seg_idx / total_segments + local_idx / (total_frames_in_seg * total_segments))
                    * 90
                )
                self.progress.emit(
                    prog, f"Хвиля від {anchor.frame_id}: кадр {frame_id}/{num_frames}..."
                )

            info = h_chain.get(frame_id)
            if info:
                metric_pts = self._project_to_metric(info["H"], anchor)
                if metric_pts is not None:
                    M, _ = cv2.estimateAffine2D(self.grid_points, metric_pts)
                    if M is not None:
                        frame_affine[frame_id] = M
                        frame_valid[frame_id] = True
                        proj = GeometryTransforms.apply_affine(self.grid_points, M)
                        rmse = np.sqrt(np.mean(np.linalg.norm(proj - metric_pts, axis=1) ** 2))
                        frame_rmse[frame_id] = rmse
                        frame_matches[frame_id] = info["matches"]

    def _build_homography_chain(self, frames, anchor, anchor_feat):
        """
        Fix 4: Використання збережених frame_poses для миттєвої пропагації.
        O(1) замість O(N) GPU-викликів матчингу.
        """
        result = {anchor.frame_id: {"H": np.eye(3, dtype=np.float32), "matches": 100}}
        anchor_id = anchor.frame_id

        try:
            pose_anchor = self.database.frame_poses[anchor_id].astype(np.float64)
            if np.abs(np.linalg.det(pose_anchor)) > 1e-9:
                inv_pose_anchor = np.linalg.inv(pose_anchor)

                for frame_id in frames:
                    if not self._is_running:
                        break
                    try:
                        pose_frame = self.database.frame_poses[frame_id].astype(np.float64)
                        # Пропускаємо кадри з вироженою позою (не збережені / zeros)
                        if np.abs(np.linalg.det(pose_frame)) < 1e-9:
                            continue
                        # H від frame до anchor через збережені pose-ланцюги
                        H = (inv_pose_anchor @ pose_frame).astype(np.float32)

                        # Fix: Якщо поза ідентична якірній (frozen pose), пробуємо екстраполяцію
                        if np.allclose(H, np.eye(3), atol=1e-6) and frame_id != anchor_id:
                            # Шукаємо тренд перед якорем (для хвоста в кінці) або після (для хвоста на початку)
                            # Робимо лінійну екстраполяцію зсуву
                            step = 1 if frame_id > anchor_id else -1
                            prev_id = anchor_id - step
                            if 0 <= prev_id < self.database.get_num_frames():
                                try:
                                    p_prev = self.database.frame_poses[prev_id].astype(np.float64)
                                    # H_step = inv(anchor) @ p_prev  => рух за 1 кадр до якоря
                                    H_step = inv_pose_anchor @ p_prev
                                    # Екстраполюємо: H_extrap = H_step ^ (dist)
                                    dist = abs(frame_id - anchor_id)
                                    # Для простоти — лінійна екстраполяція трансляції (найважливіше)
                                    H = np.eye(3, dtype=np.float32)
                                    H[0, 2] = -H_step[0, 2] * dist
                                    H[1, 2] = -H_step[1, 2] * dist
                                    # logger.debug(f"Extrapolated H for frame {frame_id} from trend at {anchor_id}")
                                except Exception:
                                    pass

                        result[frame_id] = {"H": H, "matches": 50}  # 50 = estimated
                    except Exception:
                        continue
                return result
        except Exception as e:
            logger.warning(
                f"Failed to use saved poses for anchor {anchor_id}, falling back to visual: {e}"
            )

        # Fallback до візуальної пропагації (тільки якщо поз немає АБО якщо pose-chain метод не спрацював)
        if anchor_feat is None:
            logger.warning(
                f"No features for anchor {anchor_id}, visual fallback disabled for this segment."
            )
            return result

        h_cache = {anchor_id: np.eye(3, dtype=np.float32)}
        matches_cache = {anchor_id: 100}
        prev_features = anchor_feat
        prev_frame_id = anchor_id

        for frame_id in frames:
            if not self._is_running:
                break
            try:
                curr_features = self._all_features.get(frame_id)
                if curr_features is None:
                    continue

                res = self._match_pair_with_count(curr_features, prev_features)
                if res is None:
                    continue
                H_curr_to_prev, n_matches = res

                H_prev_to_anchor = h_cache[prev_frame_id]
                H_curr_to_anchor = (
                    H_prev_to_anchor.astype(np.float64) @ H_curr_to_prev.astype(np.float64)
                ).astype(np.float32)

                h_cache[frame_id] = H_curr_to_anchor
                matches_cache[frame_id] = n_matches
                result[frame_id] = {"H": H_curr_to_anchor, "matches": n_matches}

                prev_features = curr_features
                prev_frame_id = frame_id
            except Exception:
                continue
        return result

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

                # Дані та метадані версії 2.1
                grp.attrs["version"] = "2.1"
                grp.attrs["num_anchors"] = len(anchors)
                grp.attrs["anchors_json"] = json.dumps(
                    [a.to_dict() for a in anchors], ensure_ascii=False
                )
                grp.attrs["projection_json"] = json.dumps(
                    self.calibration.converter.export_metadata()
                )

                # Датасети
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
                f"Successful propagation saved to HDF5 (rev 2.1, {len(anchors)} anchors)"
            )
        finally:
            self.database._load_hot_data()
