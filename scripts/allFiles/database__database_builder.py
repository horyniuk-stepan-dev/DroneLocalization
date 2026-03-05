import cv2
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime

from src.utils.logging_utils import get_logger
from src.models.wrappers.yolo_wrapper import YOLOWrapper
from src.models.wrappers.feature_extractor import FeatureExtractor

logger = get_logger(__name__)

_FLUSH_EVERY = 100  # кадрів між flush() викликами


class DatabaseBuilder:
    """Builds HDF5 topometric database from reference video."""

    def __init__(self, output_path, config=None):
        self.output_path = output_path
        self.config = config or {}
        self.descriptor_dim = self.config.get('dinov2', {}).get('descriptor_dim', 384)
        self.db_file = None
        logger.info(f"DatabaseBuilder initialized: {output_path}, desc_dim={self.descriptor_dim}")

    def build_from_video(
        self,
        video_path: str,
        model_manager,
        progress_callback=None,
        save_keypoint_video: bool = True,
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не вдалося відкрити відео: {video_path}")

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

        if num_frames <= 0:
            cap.release()
            raise ValueError(
                "OpenCV не зміг розпізнати відео. "
                "Спробуйте переконвертувати у стандартний MP4 (H.264)."
            )

        logger.info(f"Video: {width}x{height}, {num_frames} frames, {fps:.2f} FPS")

        # Keypoint visualization video
        kp_writer = None
        kp_video_path = None
        if save_keypoint_video:
            kp_video_path = str(Path(self.output_path).with_suffix('')) + '_keypoints.mp4'
            kp_writer = cv2.VideoWriter(
                kp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
            )
            if not kp_writer.isOpened():
                kp_writer.release()
                kp_writer = None
                logger.warning("Failed to initialize keypoint video writer — skipping")
            else:
                logger.info(f"Keypoint video: {kp_video_path}")

        # Load models
        yolo_wrapper = YOLOWrapper(model_manager.load_yolo(), model_manager.device)
        feature_extractor = FeatureExtractor(
            model_manager.load_superpoint(),
            model_manager.load_dinov2(),
            model_manager.device,
            config=self.config,
        )
        logger.success("Models loaded")

        # Create HDF5 — single open, write mode only once
        self._create_hdf5_structure(num_frames, width, height)

        current_pose  = np.eye(3, dtype=np.float32)
        prev_features = None
        processed     = 0

        try:
            self.db_file = h5py.File(self.output_path, 'a')

            for i in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Frame {i}: read failed — stopping early")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                static_mask, _ = yolo_wrapper.detect_and_mask(frame_rgb)
                features = feature_extractor.extract_features(frame_rgb, static_mask)

                if kp_writer is not None:
                    kp_writer.write(
                        self._draw_keypoints_frame(frame, features['keypoints'], static_mask, i, num_frames)
                    )

                # Accumulate pose: H(frame_i → frame_0)
                if i == 0 or prev_features is None:
                    current_pose = np.eye(3, dtype=np.float32)
                else:
                    H_step = self._compute_inter_frame_H(prev_features, features)
                    if H_step is not None:
                        current_pose = (current_pose.astype(np.float64)
                                        @ H_step.astype(np.float64)).astype(np.float32)
                    else:
                        logger.warning(f"Frame {i}: inter-frame match failed — reusing previous pose")

                prev_features = features
                self._save_frame_data(i, features, current_pose)
                processed += 1

                # Periodic flush — survive crashes
                if processed % _FLUSH_EVERY == 0:
                    self.db_file.flush()
                    logger.info(f"Flushed HDF5 | {processed}/{num_frames} frames")

                if progress_callback:
                    progress_callback(int(processed / num_frames * 100))

        except Exception as e:
            logger.error(f"Database build failed at frame {processed}: {e}", exc_info=True)
            raise
        finally:
            if self.db_file:
                self.db_file.flush()
                self.db_file.close()
            if kp_writer is not None:
                kp_writer.release()
                logger.success(f"Keypoint video saved: {kp_video_path}")
            cap.release()
            if progress_callback:
                progress_callback(100)

        logger.success(f"Database built: {self.output_path} ({processed}/{num_frames} frames)")

    def _create_hdf5_structure(self, num_frames: int, width: int, height: int) -> None:
        """Create HDF5 file structure. Overwrites existing file."""
        with h5py.File(self.output_path, 'w') as f:
            g = f.create_group('global_descriptors')
            g.create_dataset('descriptors',  shape=(num_frames, self.descriptor_dim), dtype='float32', compression='gzip')
            g.create_dataset('frame_poses',  shape=(num_frames, 3, 3),                dtype='float32', compression='gzip')
            f.create_group('local_features')
            meta = f.create_group('metadata')
            meta.attrs.update({
                'num_frames':     num_frames,
                'creation_date':  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'frame_width':    width,
                'frame_height':   height,
                'descriptor_dim': self.descriptor_dim,
            })
        logger.success(f"HDF5 structure created: {num_frames} frames, dim={self.descriptor_dim}")

    def _save_frame_data(self, frame_id: int, features: dict, pose: np.ndarray) -> None:
        self.db_file['global_descriptors']['descriptors'][frame_id] = features['global_desc']
        self.db_file['global_descriptors']['frame_poses'][frame_id] = pose

        grp = self.db_file['local_features'].create_group(f'frame_{frame_id}')
        grp.create_dataset('keypoints',   data=features['keypoints'],   dtype='float32', compression='gzip')
        grp.create_dataset('descriptors', data=features['descriptors'], dtype='float32', compression='gzip')
        # coords_2d removed — duplicate of keypoints

    def _compute_inter_frame_H(
        self,
        fa: dict,
        fb: dict,
        min_matches: int = 15,
        ransac_thresh: float = 3.0,
    ) -> np.ndarray | None:
        """Compute H(fb → fa) via brute-force descriptor matching + RANSAC."""
        kpts_a, desc_a = fa['keypoints'], fa['descriptors']
        kpts_b, desc_b = fb['keypoints'], fb['descriptors']

        if len(kpts_a) < min_matches or len(kpts_b) < min_matches:
            return None

        desc_a_n = desc_a / (np.linalg.norm(desc_a, axis=1, keepdims=True) + 1e-8)
        desc_b_n = desc_b / (np.linalg.norm(desc_b, axis=1, keepdims=True) + 1e-8)

        sim = desc_b_n @ desc_a_n.T  # (M, N) cosine similarity

        # Top-2 via argpartition — O(N) vs O(N log N) for full argsort
        top2 = np.argpartition(-sim, kth=min(2, sim.shape[1] - 1), axis=1)[:, :2]
        order = np.argsort(-sim[np.arange(len(desc_b))[:, None], top2], axis=1)
        top2 = top2[np.arange(len(desc_b))[:, None], order]

        best   = sim[np.arange(len(desc_b)), top2[:, 0]]
        second = sim[np.arange(len(desc_b)), top2[:, 1]]

        # Lowe's ratio test in distance space (1 - cosine similarity)
        # Перевід косинусної подібності в точну L2-відстань
        dist_best = np.sqrt(np.maximum(2.0 - 2.0 * best, 0.0))
        dist_second = np.sqrt(np.maximum(2.0 - 2.0 * second, 0.0))
        valid = (dist_best / (dist_second + 1e-8)) < 0.83

        if valid.sum() < min_matches:
            return None

        mkpts_b = kpts_b[valid].reshape(-1, 1, 2).astype(np.float32)
        mkpts_a = kpts_a[top2[valid, 0]].reshape(-1, 1, 2).astype(np.float32)

        H, mask = cv2.findHomography(mkpts_b, mkpts_a, cv2.USAC_MAGSAC, ransac_thresh)
        if H is None or int(np.sum(mask)) < min_matches:
            return None
        return H.astype(np.float32)

    def _draw_keypoints_frame(
        self,
        frame_bgr: np.ndarray,
        keypoints: np.ndarray,
        static_mask: np.ndarray,
        frame_id: int,
        total_frames: int,
    ) -> np.ndarray:
        vis = frame_bgr.copy()

        if static_mask is not None and (static_mask == 0).any():
            overlay = vis.copy()
            overlay[static_mask == 0] = (0, 0, 200)
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

        for x, y in keypoints:
            cx, cy = int(round(x)), int(round(y))
            cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)
            cv2.circle(vis, (cx, cy), 4, (0, 180, 0), 1)

        info_lines = [
            f"Frame: {frame_id:05d} / {total_frames:05d}",
            f"Keypoints: {len(keypoints)}",
        ]
        panel_h = len(info_lines) * 28 + 14
        cv2.rectangle(vis, (0, 0), (280, panel_h), (0, 0, 0), -1)
        for idx, line in enumerate(info_lines):
            cv2.putText(vis, line, (8, 22 + idx * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
        return vis
