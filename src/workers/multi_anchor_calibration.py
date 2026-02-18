import numpy as np
import h5py
from PyQt6.QtCore import QThread, pyqtSignal

from src.geometry.transformations import GeometryTransforms
from src.geometry.coordinates import CoordinateConverter
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CalibrationPropagationWorker(QThread):
    """
    Хвильова пропагація GPS від декількох якорів на всі кадри бази.

    Алгоритм:
    ─────────
    1. Якорі відсортовані за frame_id: [A0, A1, A2, ...]
    2. Для кожного сегменту між двома сусідніми якорями [A_i ... A_{i+1}]:
       - Хвиля від A_i іде вправо, від A_{i+1} іде вліво.
       - Кожен кадр frame_k у сегменті отримує:
             H(frame_k → A_i)   через ланцюжок від A_i
             H(frame_k → A_{i+1}) через ланцюжок від A_{i+1}
       - GPS блендується з вагою, пропорційною до оберненої відстані.
    3. Кадри лівіше A0 або правіше A_last — тільки один якір без блендінгу.
    4. Результати зберігаються у HDF5:
         calibration/h_to_anchor   — (N, 3, 3) H до найближчого якоря
         calibration/nearest_anchor_frame_id — (N,) frame_id якоря
         calibration/frame_gps     — (N, 2) фінальний (lat, lon)
         calibration/frame_valid   — (N,) bool
         calibration/anchors_json  — серіалізовані якорі
    """

    progress = pyqtSignal(int, str)
    completed = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, database, calibration, matcher, config=None):
        super().__init__()
        self.database = database
        self.calibration = calibration  # MultiAnchorCalibration
        self.matcher = matcher
        self.config = config or {}
        self._is_running = True

        self.min_matches = self.config.get('localization', {}).get('min_matches', 15)
        self.ransac_thresh = self.config.get('localization', {}).get('ransac_threshold', 3.0)

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            self._propagate()
        except Exception as e:
            logger.error(f"CalibrationPropagationWorker failed: {e}", exc_info=True)
            self.error.emit(str(e))

    # ──────────────────────────────────────────────────────────────────
    # Головний метод
    # ──────────────────────────────────────────────────────────────────

    def _propagate(self):
        num_frames = self.database.get_num_frames()
        anchors = self.calibration.anchors   # відсортовані за frame_id

        if not anchors:
            self.error.emit("Немає якорів калібрування")
            return

        frame_w = self.database.metadata.get('frame_width', 1920)
        frame_h = self.database.metadata.get('frame_height', 1080)
        cx, cy = frame_w / 2.0, frame_h / 2.0
        center = np.array([[cx, cy]], dtype=np.float32)

        logger.info(
            f"Starting multi-anchor propagation: {num_frames} frames, "
            f"{len(anchors)} anchors at frames {[a.frame_id for a in anchors]}"
        )

        # Масиви результатів
        h_to_anchor = np.zeros((num_frames, 3, 3), dtype=np.float32)
        nearest_anchor_fid = np.full(num_frames, -1, dtype=np.int32)
        frame_gps = np.zeros((num_frames, 2), dtype=np.float64)
        frame_valid = np.zeros(num_frames, dtype=bool)

        # Завантажуємо фічі якорів один раз
        self.progress.emit(0, "Завантаження фіч якорів...")
        anchor_features = {}
        for anchor in anchors:
            try:
                anchor_features[anchor.frame_id] = self.database.get_local_features(anchor.frame_id)
                logger.info(f"Anchor {anchor.frame_id}: "
                            f"{len(anchor_features[anchor.frame_id]['keypoints'])} kpts")
            except Exception as e:
                self.error.emit(f"Не вдалося завантажити якір {anchor.frame_id}: {e}")
                return

        # GPS центру кожного якоря (identity — H=eye)
        for anchor in anchors:
            h_to_anchor[anchor.frame_id] = np.eye(3, dtype=np.float32)
            nearest_anchor_fid[anchor.frame_id] = anchor.frame_id
            try:
                lat, lon = anchor.transform_to_gps(cx, cy)
                frame_gps[anchor.frame_id] = [lat, lon]
                frame_valid[anchor.frame_id] = True
                logger.info(f"Anchor {anchor.frame_id} GPS center: ({lat:.6f}, {lon:.6f})")
            except Exception as e:
                logger.error(f"Anchor {anchor.frame_id} GPS failed: {e}")

        # ── Сегменти між якорями ──────────────────────────────────────
        segments = self._build_segments(anchors, num_frames)
        total_segments = len(segments)

        for seg_idx, segment in enumerate(segments):
            if not self._is_running:
                return

            self._process_segment(
                segment=segment,
                anchor_features=anchor_features,
                center=center,
                h_to_anchor=h_to_anchor,
                nearest_anchor_fid=nearest_anchor_fid,
                frame_gps=frame_gps,
                frame_valid=frame_valid,
                seg_idx=seg_idx,
                total_segments=total_segments,
                num_frames=num_frames
            )

        # ── Зберігаємо у HDF5 ─────────────────────────────────────────
        valid_count = int(np.sum(frame_valid))
        logger.info(f"Propagation done: {valid_count}/{num_frames} frames have GPS")

        self.progress.emit(98, "Збереження у базу даних...")
        self._save_to_hdf5(h_to_anchor, nearest_anchor_fid, frame_gps, frame_valid)

        # Оновлюємо hot-дані у database loader
        self.database.h_to_anchor = h_to_anchor
        self.database.nearest_anchor_fid = nearest_anchor_fid
        self.database.frame_gps = frame_gps
        self.database.frame_valid = frame_valid

        self.progress.emit(100,
            f"Готово! {valid_count}/{num_frames} кадрів отримали GPS координати.")
        logger.success(f"Multi-anchor GPS propagation complete: "
                       f"{valid_count}/{num_frames} frames")
        self.completed.emit()

    # ──────────────────────────────────────────────────────────────────
    # Побудова сегментів
    # ──────────────────────────────────────────────────────────────────

    def _build_segments(self, anchors: list, num_frames: int) -> list:
        """
        Будуємо список сегментів для обробки.
        Кожен сегмент — dict з інформацією про діапазон кадрів і якорі.

        Типи сегментів:
          'left_tail'  — від 0 до першого якоря (тільки правий якір)
          'between'    — між двома якорями (блендінг)
          'right_tail' — від останнього якоря до кінця (тільки лівий якір)
        """
        segments = []

        # Лівий хвіст: кадри 0..anchors[0].frame_id-1
        if anchors[0].frame_id > 0:
            segments.append({
                'type': 'tail',
                'frames': list(range(anchors[0].frame_id - 1, -1, -1)),  # від якоря ліворуч
                'anchor': anchors[0],
                'direction': 'left'
            })

        # Сегменти між якорями
        for i in range(len(anchors) - 1):
            left_anchor = anchors[i]
            right_anchor = anchors[i + 1]
            mid = (left_anchor.frame_id + right_anchor.frame_id) // 2

            # Ліва половина сегменту: від left_anchor вправо до mid
            left_frames = list(range(left_anchor.frame_id + 1, mid + 1))
            # Права половина сегменту: від right_anchor вліво до mid+1
            right_frames = list(range(right_anchor.frame_id - 1, mid, -1))

            segments.append({
                'type': 'between',
                'left_anchor': left_anchor,
                'right_anchor': right_anchor,
                'left_frames': left_frames,   # обробляти від left_anchor
                'right_frames': right_frames, # обробляти від right_anchor
            })

        # Правий хвіст: кадри anchors[-1].frame_id+1..num_frames-1
        if anchors[-1].frame_id < num_frames - 1:
            segments.append({
                'type': 'tail',
                'frames': list(range(anchors[-1].frame_id + 1, num_frames)),
                'anchor': anchors[-1],
                'direction': 'right'
            })

        return segments

    # ──────────────────────────────────────────────────────────────────
    # Обробка сегменту
    # ──────────────────────────────────────────────────────────────────

    def _process_segment(self, segment, anchor_features, center,
                         h_to_anchor, nearest_anchor_fid,
                         frame_gps, frame_valid,
                         seg_idx, total_segments, num_frames):
        """Обробляє один сегмент — хвиля від якоря(ів)"""

        if segment['type'] == 'tail':
            anchor = segment['anchor']
            frames = segment['frames']
            logger.info(
                f"Segment {seg_idx+1}/{total_segments}: tail, "
                f"{len(frames)} frames from anchor {anchor.frame_id}"
            )
            self._wave_from_anchor(
                frames=frames,
                anchor=anchor,
                anchor_feat=anchor_features[anchor.frame_id],
                h_to_anchor=h_to_anchor,
                nearest_anchor_fid=nearest_anchor_fid,
                frame_gps=frame_gps,
                frame_valid=frame_valid,
                center=center,
                seg_idx=seg_idx,
                total_segments=total_segments,
                num_frames=num_frames
            )

        elif segment['type'] == 'between':
            left_anchor = segment['left_anchor']
            right_anchor = segment['right_anchor']
            left_frames = segment['left_frames']
            right_frames = segment['right_frames']

            logger.info(
                f"Segment {seg_idx+1}/{total_segments}: between "
                f"anchor {left_anchor.frame_id} ←→ anchor {right_anchor.frame_id}"
            )

            # Будуємо H від лівого якоря вправо
            h_from_left = {}   # frame_id → H(frame → left_anchor)
            self._wave_from_anchor(
                frames=left_frames,
                anchor=left_anchor,
                anchor_feat=anchor_features[left_anchor.frame_id],
                h_to_anchor=h_to_anchor,
                nearest_anchor_fid=nearest_anchor_fid,
                frame_gps=frame_gps,
                frame_valid=frame_valid,
                center=center,
                seg_idx=seg_idx,
                total_segments=total_segments,
                num_frames=num_frames,
                h_cache_out=h_from_left,
                blend_anchor_left=left_anchor,
                blend_anchor_right=right_anchor,
                anchor_feat_right=anchor_features[right_anchor.frame_id]
            )

            # Будуємо H від правого якоря вліво
            h_from_right = {}  # frame_id → H(frame → right_anchor)
            self._wave_from_anchor(
                frames=right_frames,
                anchor=right_anchor,
                anchor_feat=anchor_features[right_anchor.frame_id],
                h_to_anchor=h_to_anchor,
                nearest_anchor_fid=nearest_anchor_fid,
                frame_gps=frame_gps,
                frame_valid=frame_valid,
                center=center,
                seg_idx=seg_idx,
                total_segments=total_segments,
                num_frames=num_frames,
                h_cache_out=h_from_right,
                blend_anchor_left=left_anchor,
                blend_anchor_right=right_anchor,
                anchor_feat_left=anchor_features[left_anchor.frame_id]
            )

    def _wave_from_anchor(
        self, frames, anchor, anchor_feat,
        h_to_anchor, nearest_anchor_fid, frame_gps, frame_valid,
        center, seg_idx, total_segments, num_frames,
        h_cache_out=None,
        blend_anchor_left=None, blend_anchor_right=None,
        anchor_feat_left=None, anchor_feat_right=None
    ):
        """
        Хвиля від одного якоря вздовж списку frames.
        Кожен наступний кадр зіставляється з попереднім відомим.
        При наявності двох якорів — виконує блендінг.
        """
        # Кеш H(frame → anchor) для цієї хвилі
        # Починаємо від самого якоря (identity)
        h_wave_cache = {anchor.frame_id: np.eye(3, dtype=np.float32)}
        prev_features = anchor_feat
        prev_frame_id = anchor.frame_id

        total_frames_in_seg = len(frames)

        for local_idx, frame_id in enumerate(frames):
            if not self._is_running:
                return

            # Прогрес
            global_progress = int(
                (seg_idx / total_segments + local_idx / (total_frames_in_seg * total_segments)) * 95
            )
            if local_idx % 30 == 0:
                self.progress.emit(
                    global_progress,
                    f"Якір {anchor.frame_id}: кадр {frame_id}/{num_frames}..."
                )

            try:
                curr_features = self.database.get_local_features(frame_id)
            except Exception as e:
                logger.debug(f"Frame {frame_id}: load failed: {e}")
                continue

            # Зіставляємо з попереднім відомим кадром
            H_curr_to_prev = self._match_pair(curr_features, prev_features)

            if H_curr_to_prev is None:
                # Спробуємо midpoint між prev і curr
                mid_id = (prev_frame_id + frame_id) // 2
                if mid_id != prev_frame_id and mid_id != frame_id:
                    H_curr_to_prev = self._match_via_midpoint(
                        curr_features, frame_id, prev_frame_id
                    )

            if H_curr_to_prev is None:
                logger.warning(f"Frame {frame_id}: chain broken from {prev_frame_id}, skipping")
                # Не оновлюємо prev — наступна спроба від того ж prev
                continue

            # H(frame → anchor) = H(prev → anchor) @ H(curr → prev)
            H_prev_to_anchor = h_wave_cache[prev_frame_id]
            H_curr_to_anchor = (
                H_prev_to_anchor.astype(np.float64) @
                H_curr_to_prev.astype(np.float64)
            ).astype(np.float32)

            h_wave_cache[frame_id] = H_curr_to_anchor
            if h_cache_out is not None:
                h_cache_out[frame_id] = H_curr_to_anchor

            # Трансформуємо центр у простір якоря
            pt_in_anchor = GeometryTransforms.apply_homography(center, H_curr_to_anchor)
            if pt_in_anchor is None or len(pt_in_anchor) == 0:
                prev_features = curr_features
                prev_frame_id = frame_id
                continue

            px, py = float(pt_in_anchor[0][0]), float(pt_in_anchor[0][1])

            # ── Блендінг між двома якорями ────────────────────────────
            if blend_anchor_left is not None and blend_anchor_right is not None:
                # Визначаємо GPS через поточний якір
                try:
                    mx1, my1 = anchor.pixel_to_metric(px, py)
                except Exception:
                    prev_features = curr_features
                    prev_frame_id = frame_id
                    continue

                # Шукаємо GPS через другий якір (якщо маємо його H)
                other_anchor = (blend_anchor_right
                                if anchor.frame_id == blend_anchor_left.frame_id
                                else blend_anchor_left)
                other_feat = (anchor_feat_right if anchor_feat_right is not None
                              else anchor_feat_left)

                # GPS via other anchor — беремо з вже обчислених frame_gps
                # якщо є, або пропускаємо блендінг
                other_gps = None
                if frame_valid[frame_id]:
                    # Вже є від іншої хвилі
                    other_lat, other_lon = frame_gps[frame_id]
                    other_mx, other_my = CoordinateConverter.gps_to_metric(other_lat, other_lon)
                    other_gps = (other_mx, other_my)

                if other_gps is not None:
                    # Блендінг між двома результатами
                    blended = self.calibration.blend_metric(
                        frame_id,
                        (mx1, my1),
                        other_gps,
                        blend_anchor_left,
                        blend_anchor_right
                    )
                    lat, lon = CoordinateConverter.metric_to_gps(blended[0], blended[1])
                else:
                    lat, lon = CoordinateConverter.metric_to_gps(mx1, my1)

            else:
                # Тільки один якір (хвіст)
                try:
                    mx, my = anchor.pixel_to_metric(px, py)
                    lat, lon = CoordinateConverter.metric_to_gps(mx, my)
                except Exception as e:
                    logger.debug(f"Frame {frame_id}: metric failed: {e}")
                    prev_features = curr_features
                    prev_frame_id = frame_id
                    continue

            # Зберігаємо результат
            h_to_anchor[frame_id] = H_curr_to_anchor
            nearest_anchor_fid[frame_id] = anchor.frame_id
            frame_gps[frame_id] = [lat, lon]
            frame_valid[frame_id] = True

            prev_features = curr_features
            prev_frame_id = frame_id

    # ──────────────────────────────────────────────────────────────────
    # Допоміжні методи
    # ──────────────────────────────────────────────────────────────────

    def _match_pair(self, features_a: dict, features_b: dict) -> np.ndarray | None:
        """H(a → b) або None"""
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
        except Exception as e:
            logger.debug(f"Match pair failed: {e}")
            return None

    def _match_via_midpoint(
        self, curr_features: dict, curr_id: int, prev_id: int
    ) -> np.ndarray | None:
        """Спроба зіставлення через проміжний кадр посередині"""
        mid_id = (curr_id + prev_id) // 2
        if mid_id == curr_id or mid_id == prev_id:
            return None
        try:
            mid_features = self.database.get_local_features(mid_id)
            H1 = self._match_pair(curr_features, mid_features)
            prev_features = self.database.get_local_features(prev_id)
            H2 = self._match_pair(mid_features, prev_features)
            if H1 is None or H2 is None:
                return None
            return (H2.astype(np.float64) @ H1.astype(np.float64)).astype(np.float32)
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────
    # Збереження у HDF5
    # ──────────────────────────────────────────────────────────────────

    def _save_to_hdf5(self, h_to_anchor, nearest_anchor_fid, frame_gps, frame_valid):
        import json as _json

        db_path = self.database.db_path
        logger.info(f"Saving multi-anchor propagation to {db_path}...")

        self.database.close()
        try:
            with h5py.File(db_path, 'a') as f:
                if 'calibration' in f:
                    del f['calibration']

                grp = f.create_group('calibration')
                grp.create_dataset('h_to_anchor', data=h_to_anchor,
                                   dtype='float32', compression='gzip')
                grp.create_dataset('nearest_anchor_frame_id',
                                   data=nearest_anchor_fid,
                                   dtype='int32', compression='gzip')
                grp.create_dataset('frame_gps', data=frame_gps,
                                   dtype='float64', compression='gzip')
                grp.create_dataset('frame_valid',
                                   data=frame_valid.astype(np.uint8),
                                   compression='gzip')

                # Зберігаємо якорі як JSON рядок
                anchors_json = _json.dumps(
                    [a.to_dict() for a in self.calibration.anchors]
                )
                grp.attrs['anchors_json'] = anchors_json
                grp.attrs['num_anchors'] = len(self.calibration.anchors)

            logger.success("Multi-anchor propagation data saved to HDF5")
        finally:
            self.database._load_hot_data()