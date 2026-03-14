import cv2
import numpy as np
import h5py
import json
from PyQt6.QtCore import QThread, pyqtSignal

from src.geometry.transformations import GeometryTransforms
from src.geometry.coordinates import CoordinateConverter
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

        self.min_matches = self.config.get('localization', {}).get('min_matches', 15)
        self.ransac_thresh = self.config.get('localization', {}).get('ransac_threshold', 3.0)

        self.frame_w = self.database.metadata.get('frame_width', 1920)
        self.frame_h = self.database.metadata.get('frame_height', 1080)

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
        anchors = sorted(self.calibration.anchors, key=lambda a: a.frame_id)

        if not anchors:
            self.error.emit("Немає якорів калібрування")
            return

        logger.info(f"Starting visual wave propagation for {num_frames} frames using {len(anchors)} anchors")

        frame_affine = np.zeros((num_frames, 2, 3), dtype=np.float32)
        frame_valid = np.zeros(num_frames, dtype=bool)
        
        # QA metrics
        frame_rmse = np.zeros(num_frames, dtype=np.float32)
        frame_disagreement = np.zeros(num_frames, dtype=np.float32)

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
                anchor_features[anchor.frame_id] = self._all_features[anchor.frame_id]
                frame_affine[anchor.frame_id] = anchor.affine_matrix
                frame_valid[anchor.frame_id] = True
                frame_rmse[anchor.frame_id] = getattr(anchor, 'rmse_m', 0.0)
            except Exception as e:
                self.error.emit(f"Не вдалося завантажити якір {anchor.frame_id}: {e}")
                return

        segments = self._build_segments(anchors, num_frames)
        total_segments = len(segments)

        for seg_idx, segment in enumerate(segments):
            if not self._is_running:
                return
            self._process_segment(
                segment=segment,
                anchor_features=anchor_features,
                frame_affine=frame_affine,
                frame_valid=frame_valid,
                frame_rmse=frame_rmse,
                frame_disagreement=frame_disagreement,
                seg_idx=seg_idx,
                total_segments=total_segments,
                num_frames=num_frames
            )

        valid_count = int(np.sum(frame_valid))
        self.progress.emit(90, "Збереження метрик якості у HDF5...")
        self._save_to_hdf5(frame_affine, frame_valid, frame_rmse, frame_disagreement, anchors)

        self.progress.emit(100, f"Готово! {valid_count}/{num_frames} кадрів отримали координати.")
        self.completed.emit()

    def _build_segments(self, anchors: list, num_frames: int) -> list:
        segments = []
        if anchors[0].frame_id > 0:
            segments.append({
                'type': 'tail',
                'frames': list(range(anchors[0].frame_id - 1, -1, -1)),
                'anchor': anchors[0]
            })

        for i in range(len(anchors) - 1):
            left_anchor = anchors[i]
            right_anchor = anchors[i + 1]
            segments.append({
                'type': 'between',
                'left_anchor': left_anchor,
                'right_anchor': right_anchor,
                'frames': list(range(left_anchor.frame_id + 1, right_anchor.frame_id))
            })

        if anchors[-1].frame_id < num_frames - 1:
            segments.append({
                'type': 'tail',
                'frames': list(range(anchors[-1].frame_id + 1, num_frames)),
                'anchor': anchors[-1]
            })

        return segments

    def _process_segment(self, segment, anchor_features, frame_affine, frame_valid, 
                         frame_rmse, frame_disagreement, seg_idx, total_segments,
                         num_frames):
        if segment['type'] == 'tail':
            anchor = segment['anchor']
            frames = segment['frames']
            self._wave_from_anchor(
                frames=frames, anchor=anchor, anchor_feat=anchor_features[anchor.frame_id],
                frame_affine=frame_affine, frame_valid=frame_valid, frame_rmse=frame_rmse,
                seg_idx=seg_idx, total_segments=total_segments, num_frames=num_frames
            )
        elif segment['type'] == 'between':
            left_anchor = segment['left_anchor']
            right_anchor = segment['right_anchor']
            frames = segment['frames']

            h_from_left = self._build_homography_chain(frames, left_anchor, anchor_features[left_anchor.frame_id])
            frames_reversed = list(reversed(frames))
            h_from_right = self._build_homography_chain(frames_reversed, right_anchor,
                                                         anchor_features[right_anchor.frame_id])

            total_frames_in_seg = len(frames)
            for local_idx, frame_id in enumerate(frames):
                if not self._is_running: return

                if local_idx % 10 == 0:
                    prog = int((seg_idx / total_segments + local_idx / (total_frames_in_seg * total_segments)) * 90)
                    self.progress.emit(prog, f"Блендінг: кадр {frame_id}/{num_frames}...")

                H_to_left = h_from_left.get(frame_id)
                H_to_right = h_from_right.get(frame_id)

                metric_pts_left = self._project_to_metric(H_to_left, left_anchor) if H_to_left is not None else None
                metric_pts_right = self._project_to_metric(H_to_right, right_anchor) if H_to_right is not None else None

                final_metric_pts = None
                disagreement = 0.0

                if metric_pts_left is not None and metric_pts_right is not None:
                    # Обчислюємо disagreement (розбіжність) у метрах
                    disagreement = np.mean(np.linalg.norm(metric_pts_left - metric_pts_right, axis=1))
                    
                    dist_to_left = abs(frame_id - left_anchor.frame_id)
                    dist_to_right = abs(frame_id - right_anchor.frame_id)
                    weight_left = dist_to_right / (dist_to_left + dist_to_right)
                    weight_right = 1.0 - weight_left
                    final_metric_pts = metric_pts_left * weight_left + metric_pts_right * weight_right
                elif metric_pts_left is not None:
                    final_metric_pts = metric_pts_left
                elif metric_pts_right is not None:
                    final_metric_pts = metric_pts_right

                if final_metric_pts is not None:
                    M, _ = cv2.estimateAffine2D(self.grid_points, final_metric_pts)
                    if M is not None:
                        frame_affine[frame_id] = M
                        frame_valid[frame_id] = True
                        
                        # Обчислюємо RMSE похибку афінної апроксимації сітки
                        proj = GeometryTransforms.apply_affine(self.grid_points, M)
                        rmse = np.sqrt(np.mean(np.linalg.norm(proj - final_metric_pts, axis=1)**2))
                        frame_rmse[frame_id] = rmse
                        frame_disagreement[frame_id] = disagreement

    def _wave_from_anchor(self, frames, anchor, anchor_feat, frame_affine, frame_valid, frame_rmse,
                          seg_idx, total_segments, num_frames):
        h_chain = self._build_homography_chain(frames, anchor, anchor_feat)
        total_frames_in_seg = len(frames)

        for local_idx, frame_id in enumerate(frames):
            if not self._is_running: return

            if local_idx % 20 == 0:
                prog = int((seg_idx / total_segments + local_idx / (total_frames_in_seg * total_segments)) * 90)
                self.progress.emit(prog, f"Хвиля від {anchor.frame_id}: кадр {frame_id}/{num_frames}...")

            H_to_anchor = h_chain.get(frame_id)
            if H_to_anchor is not None:
                metric_pts = self._project_to_metric(H_to_anchor, anchor)
                if metric_pts is not None:
                    M, _ = cv2.estimateAffine2D(self.grid_points, metric_pts)
                    if M is not None:
                        frame_affine[frame_id] = M
                        frame_valid[frame_id] = True
                        
                        # Обчислюємо RMSE похибку афінної апроксимації сітки
                        proj = GeometryTransforms.apply_affine(self.grid_points, M)
                        rmse = np.sqrt(np.mean(np.linalg.norm(proj - metric_pts, axis=1)**2))
                        frame_rmse[frame_id] = rmse

    def _get_stored_relative_H(self, frame_from: int, frame_to: int):
        try:
            pose_from = self.database.frame_poses[frame_from]
            pose_to = self.database.frame_poses[frame_to]
            H = np.linalg.inv(pose_to.astype(np.float64)) @ pose_from.astype(np.float64)
            return H.astype(np.float32)
        except (np.linalg.LinAlgError, IndexError):
            return None

    def _build_homography_chain(self, frames, anchor, anchor_feat):
        result = {}
        anchor_id = anchor.frame_id
        pose_anchor = None
        try:
            pose_anchor = self.database.frame_poses[anchor_id].astype(np.float64)
            inv_pose_anchor = np.linalg.inv(pose_anchor)
        except (IndexError, np.linalg.LinAlgError):
            inv_pose_anchor = None

        if inv_pose_anchor is not None:
            # RELOCK STRATEGY: Periodic re-matching to anchor for high precision 
            # if the frame is close enough (e.g. within 100 frames)
            relock_interval = 50
            
            for i, frame_id in enumerate(frames):
                if not self._is_running: break
                
                try:
                    # 1. Base estimation via stored poses
                    pose_frame = self.database.frame_poses[frame_id].astype(np.float64)
                    H = inv_pose_anchor @ pose_frame
                    
                    # 2. Relock every N frames using direct match to anchor features 
                    # (helps if poses drift over time or if they were extracted with small errors)
                    # For now, we use the poses as primary source if available.
                    result[frame_id] = H
                except (IndexError, np.linalg.LinAlgError):
                    continue
        else:
            # Fallback path (no poses in DB)
            h_cache = {anchor_id: np.eye(3, dtype=np.float32)}
            prev_features = anchor_feat
            prev_frame_id = anchor_id

            for frame_id in frames:
                if not self._is_running: break
                try:
                    curr_features = self._all_features.get(frame_id)
                    if curr_features is None: continue
                    H_curr_to_prev = self._match_pair(curr_features, prev_features)
                    if H_curr_to_prev is None: continue

                    H_prev_to_anchor = h_cache[prev_frame_id]
                    H_curr_to_anchor = (H_prev_to_anchor.astype(np.float64) @ H_curr_to_prev.astype(np.float64)).astype(np.float32)

                    h_cache[frame_id] = H_curr_to_anchor
                    result[frame_id] = H_curr_to_anchor

                    prev_features = curr_features
                    prev_frame_id = frame_id
                except Exception:
                    continue
        return result

    def _project_to_metric(self, H_to_anchor, anchor):
        pts_in_anchor = GeometryTransforms.apply_homography(self.grid_points, H_to_anchor)
        if pts_in_anchor is None or len(pts_in_anchor) != len(self.grid_points):
            return None
        metric_pts = GeometryTransforms.apply_affine(pts_in_anchor, anchor.affine_matrix)
        return metric_pts

    def _match_pair(self, features_a: dict, features_b: dict) -> np.ndarray | None:
        try:
            mkpts_a, mkpts_b = self.matcher.match(features_a, features_b)
            if len(mkpts_a) < self.min_matches:
                return None
            H, mask = GeometryTransforms.estimate_homography(
                mkpts_a, mkpts_b, ransac_threshold=self.ransac_thresh,
                max_iters=1000, confidence=0.995
            )
            if H is None or int(np.sum(mask)) < self.min_matches:
                return None
            return H
        except Exception:
            return None

    def _save_to_hdf5(self, frame_affine, frame_valid, frame_rmse, frame_disagreement, anchors):
        db_path = self.database.db_path
        self.database.close()
        try:
            with h5py.File(db_path, 'a') as f:
                if 'calibration' in f:
                    del f['calibration']
                grp = f.create_group('calibration')
                # Основні дані
                grp.create_dataset('frame_affine', data=frame_affine, dtype='float32', compression='gzip')
                grp.create_dataset('frame_valid', data=frame_valid.astype(np.uint8), compression='gzip')
                
                # Метрики якості (QA)
                grp.create_dataset('frame_rmse', data=frame_rmse, dtype='float32', compression='gzip')
                grp.create_dataset('frame_disagreement', data=frame_disagreement, dtype='float32', compression='gzip')
                
                anchors_json = json.dumps([a.to_dict() for a in anchors])
                grp.attrs['anchors_json'] = anchors_json
                grp.attrs['num_anchors'] = len(anchors)
                grp.attrs['propagation_version'] = '2.0-grid'
                
                if hasattr(self.calibration, 'reference_gps') and self.calibration.reference_gps:
                    grp.attrs['reference_gps'] = json.dumps(self.calibration.reference_gps)
            logger.success("Quality-aware propagation metrics saved to HDF5")
        finally:
            self.database._load_hot_data()