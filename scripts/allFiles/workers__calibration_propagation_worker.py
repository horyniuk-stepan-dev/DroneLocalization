import cv2
import json
import numpy as np
import h5py
from PyQt6.QtCore import QThread, pyqtSignal

from src.geometry.transformations import GeometryTransforms
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CalibrationPropagationWorker(QThread):
    """
    Wave propagation via visual matching (LightGlue/SuperPoint).
    Generates a metric affine matrix (2×3) for each database frame.
    """

    progress  = pyqtSignal(int, str)
    completed = pyqtSignal()
    error     = pyqtSignal(str)

    def __init__(self, database, calibration, matcher, config=None):
        super().__init__()
        self.database      = database
        self.calibration   = calibration
        self.matcher       = matcher
        self.config        = config or {}
        self._is_running   = False

        self.min_matches    = self.config.get('localization', {}).get('min_matches', 15)
        self.ransac_thresh  = self.config.get('localization', {}).get('ransac_threshold', 3.0)

        self.frame_w = self.database.metadata.get('frame_width', 1920)
        self.frame_h = self.database.metadata.get('frame_height', 1080)
        self.corners = np.array(
            [[0, 0], [self.frame_w, 0], [self.frame_w, self.frame_h], [0, self.frame_h]],
            dtype=np.float32,
        )

    def stop(self):
        self._is_running = False
        self.wait()

    def run(self):
        self._is_running = True
        try:
            self._propagate()
        except Exception as e:
            logger.error(f"Propagation failed: {e}", exc_info=True)
            self.error.emit(str(e))

    # ── Main pipeline ────────────────────────────────────────────────────────

    def _propagate(self):
        num_frames = self.database.get_num_frames()
        anchors = sorted(self.calibration.anchors, key=lambda a: a.frame_id)

        if not anchors:
            self.error.emit("Немає якорів калібрування")
            return

        logger.info(f"Propagation: {num_frames} frames, {len(anchors)} anchors")

        frame_affine = np.zeros((num_frames, 2, 3), dtype=np.float32)
        frame_valid  = np.zeros(num_frames, dtype=bool)

        # Seed anchors
        self.progress.emit(0, "Завантаження фіч якорів...")
        anchor_features = {}
        for anchor in anchors:
            try:
                anchor_features[anchor.frame_id] = self.database.get_local_features(anchor.frame_id)
                frame_affine[anchor.frame_id] = anchor.affine_matrix
                frame_valid[anchor.frame_id]  = True
            except Exception as e:
                self.error.emit(f"Не вдалося завантажити якір {anchor.frame_id}: {e}")
                return

        segments = self._build_segments(anchors, num_frames)

        for seg_idx, segment in enumerate(segments):
            if not self._is_running:
                logger.info("Propagation cancelled by user")
                return
            self._process_segment(
                segment, anchor_features, frame_affine, frame_valid,
                seg_idx, len(segments), num_frames,
            )

        if not self._is_running:
            return

        valid_count = int(np.sum(frame_valid))
        self.progress.emit(90, "Збереження метричних матриць у HDF5...")
        self._save_to_hdf5(frame_affine, frame_valid, anchors)

        self.progress.emit(100, f"Готово! {valid_count}/{num_frames} кадрів калібровано.")
        logger.success(f"Propagation complete: {valid_count}/{num_frames} frames valid")
        self.completed.emit()

    # ── Segment building ─────────────────────────────────────────────────────

    def _build_segments(self, anchors: list, num_frames: int) -> list:
        segments = []
        if anchors[0].frame_id > 0:
            segments.append({
                'type': 'tail',
                'frames': list(range(anchors[0].frame_id - 1, -1, -1)),
                'anchor': anchors[0],
            })
        for i in range(len(anchors) - 1):
            segments.append({
                'type': 'between',
                'left_anchor':  anchors[i],
                'right_anchor': anchors[i + 1],
                'frames': list(range(anchors[i].frame_id + 1, anchors[i + 1].frame_id)),
            })
        if anchors[-1].frame_id < num_frames - 1:
            segments.append({
                'type': 'tail',
                'frames': list(range(anchors[-1].frame_id + 1, num_frames)),
                'anchor': anchors[-1],
            })
        return segments

    # ── Segment processing ───────────────────────────────────────────────────

    def _process_segment(self, segment, anchor_features, frame_affine, frame_valid,
                         seg_idx, total_segments, num_frames):
        if segment['type'] == 'tail':
            anchor = segment['anchor']
            self._wave_from_anchor(
                frames=segment['frames'],
                anchor=anchor,
                anchor_feat=anchor_features[anchor.frame_id],
                frame_affine=frame_affine,
                frame_valid=frame_valid,
                seg_idx=seg_idx,
                total_segments=total_segments,
                num_frames=num_frames,
            )

        elif segment['type'] == 'between':
            left_a, right_a = segment['left_anchor'], segment['right_anchor']
            frames = segment['frames']

            h_from_left  = self._build_homography_chain(frames, left_a, anchor_features[left_a.frame_id])
            h_from_right = self._build_homography_chain(
                list(reversed(frames)), right_a, anchor_features[right_a.frame_id]
            )

            total_in_seg = max(len(frames), 1)
            emit_every = max(1, total_in_seg // 20)

            for local_idx, frame_id in enumerate(frames):
                if not self._is_running:
                    return

                if local_idx % emit_every == 0:
                    prog = int((seg_idx / total_segments
                                + local_idx / (total_in_seg * total_segments)) * 90)
                    self.progress.emit(prog, f"Блендінг: кадр {frame_id}/{num_frames}...")

                H_left  = h_from_left.get(frame_id)
                H_right = h_from_right.get(frame_id)
                m_left  = self._project_to_metric(H_left,  left_a)  if H_left  is not None else None
                m_right = self._project_to_metric(H_right, right_a) if H_right is not None else None

                if m_left is not None and m_right is not None:
                    d_l = abs(frame_id - left_a.frame_id)
                    d_r = abs(frame_id - right_a.frame_id)
                    w_l = d_r / (d_l + d_r)
                    final = m_left * w_l + m_right * (1.0 - w_l)
                elif m_left is not None:
                    final = m_left
                elif m_right is not None:
                    final = m_right
                else:
                    continue

                M, _ = cv2.estimateAffine2D(self.corners, final)
                if M is not None:
                    frame_affine[frame_id] = M
                    frame_valid[frame_id]  = True

    def _wave_from_anchor(self, frames, anchor, anchor_feat, frame_affine, frame_valid,
                          seg_idx, total_segments, num_frames):
        h_chain = self._build_homography_chain(frames, anchor, anchor_feat)
        total_in_seg = max(len(frames), 1)
        emit_every = max(1, total_in_seg // 20)

        for local_idx, frame_id in enumerate(frames):
            if not self._is_running:
                return

            if local_idx % emit_every == 0:
                prog = int((seg_idx / total_segments
                            + local_idx / (total_in_seg * total_segments)) * 90)
                self.progress.emit(prog, f"Хвиля від {anchor.frame_id}: кадр {frame_id}/{num_frames}...")

            H = h_chain.get(frame_id)
            if H is None:
                continue
            metric_corners = self._project_to_metric(H, anchor)
            if metric_corners is None:
                continue
            M, _ = cv2.estimateAffine2D(self.corners, metric_corners)
            if M is not None:
                frame_affine[frame_id] = M
                frame_valid[frame_id]  = True

    # ── Homography chain ─────────────────────────────────────────────────────

    def _build_homography_chain(self, frames: list, anchor, anchor_feat: dict) -> dict[int, np.ndarray]:
        """
        Build accumulated homography H(frame_i → anchor) for each frame.
        Chain sliding window: prev_features always updated (even on match failure)
        to prevent distance blow-up from a frozen reference.
        """
        H_accumulated = np.eye(3, dtype=np.float64)
        prev_features = anchor_feat
        result: dict[int, np.ndarray] = {}

        for frame_id in frames:
            if not self._is_running:
                break
            try:
                curr_features = self.database.get_local_features(frame_id)
                H_step = self._match_pair(curr_features, prev_features)

                if H_step is not None:
                    H_accumulated = H_accumulated @ H_step.astype(np.float64)
                    result[frame_id] = H_accumulated.astype(np.float32)

                # Always slide window — prevents chain freeze after one bad frame
                prev_features = curr_features

            except Exception as e:
                logger.debug(f"Chain frame {frame_id} skipped: {e}")
                continue

        return result

    # ── Geometry helpers ─────────────────────────────────────────────────────

    def _project_to_metric(self, H_to_anchor: np.ndarray, anchor) -> np.ndarray | None:
        pts = GeometryTransforms.apply_homography(self.corners, H_to_anchor)
        if pts is None or len(pts) != 4:
            return None
        return GeometryTransforms.apply_affine(pts, anchor.affine_matrix)

    def _match_pair(self, features_a: dict, features_b: dict) -> np.ndarray | None:
        try:
            mkpts_a, mkpts_b = self.matcher.match(features_a, features_b)
            if len(mkpts_a) < self.min_matches:
                return None
            H, mask = GeometryTransforms.estimate_homography(
                mkpts_a, mkpts_b, ransac_threshold=self.ransac_thresh
            )
            if H is None or int(np.sum(mask)) < self.min_matches:
                return None
            return H
        except Exception:
            return None

    # ── HDF5 persistence ─────────────────────────────────────────────────────

    def _save_to_hdf5(self, frame_affine: np.ndarray, frame_valid: np.ndarray, anchors: list) -> None:
        db_path = self.database.db_path
        self.database.close()
        try:
            with h5py.File(db_path, 'a') as f:
                if 'calibration' in f:
                    del f['calibration']
                grp = f.create_group('calibration')
                grp.create_dataset('frame_affine', data=frame_affine,
                                   dtype='float32', compression='gzip')
                grp.create_dataset('frame_valid',  data=frame_valid.astype(np.uint8),
                                   compression='gzip')
                grp.attrs['anchors_json'] = json.dumps([a.to_dict() for a in anchors])
                grp.attrs['num_anchors']  = len(anchors)
            logger.success("Propagation data saved to HDF5")
        finally:
            try:
                self.database._load_hot_data()
            except Exception as e:
                logger.error(f"Failed to reload database after save: {e}")
                self.error.emit(f"БД збережено, але не вдалося перезавантажити: {e}")
