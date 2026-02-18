import numpy as np
import h5py
from PyQt6.QtCore import QThread, pyqtSignal

from src.geometry.transformations import GeometryTransforms
from src.geometry.coordinates import CoordinateConverter
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CalibrationPropagationWorker(QThread):
    """
    Після GPS-калібрування одного кадру — розповсюджує координати
    на всі кадри бази даних.

    Алгоритм для кожного кадру i:
      1. Будуємо H(frame_i → calib_frame) через ланцюжок проміжних кадрів.
      2. Трансформуємо центр кадру у простір calib_frame.
      3. Застосовуємо affine_matrix → метричні координати → GPS.
      4. Зберігаємо H(frame_i → calib_frame) і GPS(lat, lon) у HDF5.

    Після завершення localizer просто читає готові дані — без
    обчислення ланцюжків під час реального часу.
    """

    progress = pyqtSignal(int, str)      # (percent, message)
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
        self.chain_step = self.config.get('localization', {}).get('chain_step', 10)

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            self._propagate()
        except Exception as e:
            logger.error(f"CalibrationPropagationWorker failed: {e}", exc_info=True)
            self.error.emit(str(e))

    def _propagate(self):
        num_frames = self.database.get_num_frames()
        calib_id = self.calibration.calib_frame_id

        logger.info(f"Starting GPS propagation: {num_frames} frames, calib_frame={calib_id}")
        self.progress.emit(0, f"Завантаження кадру калібрування ({calib_id})...")

        # Завантажуємо фічі кадру калібрування один раз
        calib_features = self.database.get_local_features(calib_id)
        logger.info(f"Calib frame {calib_id}: {len(calib_features['keypoints'])} keypoints")

        # Масиви результатів
        h_to_calib = np.zeros((num_frames, 3, 3), dtype=np.float32)
        frame_gps = np.zeros((num_frames, 2), dtype=np.float64)   # (lat, lon)
        frame_valid = np.zeros(num_frames, dtype=bool)

        # Кадр калібрування — тривіальний випадок (identity)
        h_to_calib[calib_id] = np.eye(3, dtype=np.float32)

        # GPS центру кадру калібрування
        frame_w = self.database.metadata.get('frame_width', 1920)
        frame_h = self.database.metadata.get('frame_height', 1080)
        cx, cy = frame_w / 2.0, frame_h / 2.0

        try:
            lat_c, lon_c = self.calibration.transform_to_gps(cx, cy)
            frame_gps[calib_id] = [lat_c, lon_c]
            frame_valid[calib_id] = True
            logger.info(f"Calib frame GPS center: ({lat_c:.6f}, {lon_c:.6f})")
        except Exception as e:
            logger.error(f"Cannot compute calib frame GPS: {e}")
            self.error.emit(f"Помилка обчислення GPS кадру калібрування: {e}")
            return

        # Кеш вже обчислених H(frame_i → calib)
        # Починаємо від calib_id і розповсюджуємося в обидва боки
        h_cache: dict[int, np.ndarray] = {calib_id: np.eye(3, dtype=np.float32)}

        # Обробляємо кадри від calib_id вперед і назад
        # Порядок: спочатку ті що поруч з calib, потім далі
        order = self._build_processing_order(calib_id, num_frames)

        for idx, frame_id in enumerate(order):
            if not self._is_running:
                logger.info("Propagation stopped by user")
                return

            if frame_id == calib_id:
                continue

            percent = int((idx + 1) / num_frames * 100)
            if idx % 20 == 0:
                self.progress.emit(
                    percent,
                    f"Розповсюдження координат: {idx + 1}/{num_frames} кадрів..."
                )

            H = self._compute_h_to_calib(frame_id, calib_id, calib_features, h_cache)

            if H is None:
                logger.warning(f"Frame {frame_id}: failed to compute H_to_calib, skipping")
                continue

            h_to_calib[frame_id] = H
            h_cache[frame_id] = H

            # GPS центру цього кадру
            center = np.array([[cx, cy]], dtype=np.float32)
            pt_in_calib = GeometryTransforms.apply_homography(center, H)
            if pt_in_calib is None or len(pt_in_calib) == 0:
                continue

            try:
                lat, lon = self.calibration.transform_to_gps(
                    float(pt_in_calib[0][0]), float(pt_in_calib[0][1])
                )
                frame_gps[frame_id] = [lat, lon]
                frame_valid[frame_id] = True
            except Exception as e:
                logger.debug(f"Frame {frame_id}: GPS transform failed: {e}")

        # Зберігаємо результати у HDF5
        valid_count = int(np.sum(frame_valid))
        logger.info(f"Propagation done: {valid_count}/{num_frames} frames have GPS")

        self.progress.emit(99, "Збереження у базу даних...")
        self._save_to_hdf5(h_to_calib, frame_gps, frame_valid, calib_id)

        # Оновлюємо hot-дані у database loader щоб localizer міг їх читати
        self.database.h_to_calib = h_to_calib
        self.database.frame_gps = frame_gps
        self.database.frame_valid = frame_valid
        self.database.calib_frame_id = calib_id

        self.progress.emit(100, f"Готово! {valid_count}/{num_frames} кадрів отримали GPS координати.")
        logger.success(f"GPS propagation complete: {valid_count}/{num_frames} frames")
        self.completed.emit()

    def _build_processing_order(self, calib_id: int, num_frames: int) -> list:
        """
        Будуємо порядок обробки від calib_id назовні в обидва боки.
        Це гарантує що h_cache заповнюється від відомих до невідомих,
        і сусідні кадри завжди мають вже обраховану H для опори.
        """
        order = []
        left = calib_id - 1
        right = calib_id + 1
        while left >= 0 or right < num_frames:
            if left >= 0:
                order.append(left)
                left -= 1
            if right < num_frames:
                order.append(right)
                right += 1
        return order

    def _compute_h_to_calib(
        self,
        frame_id: int,
        calib_id: int,
        calib_features: dict,
        h_cache: dict
    ) -> np.ndarray | None:
        """
        Обчислити H(frame_id → calib_frame).

        Стратегія:
        1. Пряме зіставлення frame_id → calib_frame.
        2. Через найближчий вже відомий кадр з h_cache:
             H(frame_id → calib) = H(frame_id → known) @ H(known → calib)
           Оскільки порядок обробки від calib назовні, найближчий сусід
           (frame_id ± 1) вже є в кеші.
        """
        try:
            curr_features = self.database.get_local_features(frame_id)
        except Exception as e:
            logger.debug(f"Cannot load frame {frame_id}: {e}")
            return None

        # Спроба 1: прямий матчинг з calib_frame
        H_direct = self._match_pair(curr_features, calib_features)
        if H_direct is not None:
            return H_direct

        # Спроба 2: через найближчий відомий кадр
        # Шукаємо найближчий frame_id у h_cache (в обох напрямках)
        nearest_id = self._find_nearest_cached(frame_id, calib_id, h_cache)
        if nearest_id is None:
            return None

        try:
            nearest_features = self.database.get_local_features(nearest_id)
        except Exception:
            return None

        H_curr_to_nearest = self._match_pair(curr_features, nearest_features)
        if H_curr_to_nearest is None:
            # Спробуємо midpoint
            mid_id = (frame_id + nearest_id) // 2
            if mid_id != frame_id and mid_id != nearest_id:
                try:
                    mid_features = self.database.get_local_features(mid_id)
                    H1 = self._match_pair(curr_features, mid_features)
                    H2 = self._match_pair(mid_features, nearest_features)
                    if H1 is not None and H2 is not None:
                        H_curr_to_nearest = (H2.astype(np.float64) @ H1.astype(np.float64)).astype(np.float32)
                except Exception:
                    pass

            if H_curr_to_nearest is None:
                return None

        # H(frame_id → calib) = H(nearest → calib) @ H(frame_id → nearest)
        H_nearest_to_calib = h_cache[nearest_id]
        H_result = (H_nearest_to_calib.astype(np.float64) @
                    H_curr_to_nearest.astype(np.float64)).astype(np.float32)
        return H_result

    def _find_nearest_cached(self, frame_id: int, calib_id: int, h_cache: dict) -> int | None:
        """Знайти найближчий frame_id у кеші"""
        # Шукаємо найближчий у бік calib_id
        step = 1 if calib_id > frame_id else -1
        for offset in range(1, abs(calib_id - frame_id) + 1):
            candidate = frame_id + offset * step
            if candidate in h_cache:
                return candidate
        return None

    def _match_pair(self, features_a: dict, features_b: dict) -> np.ndarray | None:
        """Зіставити два кадри, повернути H(a→b) або None"""
        try:
            mkpts_a, mkpts_b = self.matcher.match(features_a, features_b)
            if len(mkpts_a) < self.min_matches:
                return None
            H, mask = GeometryTransforms.estimate_homography(
                mkpts_a, mkpts_b, ransac_threshold=self.ransac_thresh
            )
            if H is None:
                return None
            if int(np.sum(mask)) < self.min_matches:
                return None
            return H
        except Exception as e:
            logger.debug(f"Match pair failed: {e}")
            return None

    def _save_to_hdf5(
        self,
        h_to_calib: np.ndarray,
        frame_gps: np.ndarray,
        frame_valid: np.ndarray,
        calib_id: int
    ):
        """Зберегти результати пропагації у HDF5 файл"""
        db_path = self.database.db_path
        logger.info(f"Saving propagation results to {db_path}...")

        # Потрібно відкрити у режимі запису — закриваємо read-only handle
        self.database.close()

        try:
            with h5py.File(db_path, 'a') as f:
                # Видаляємо старі дані пропагації якщо є
                if 'calibration' in f:
                    del f['calibration']

                grp = f.create_group('calibration')
                grp.create_dataset('h_to_calib', data=h_to_calib,
                                   dtype='float32', compression='gzip')
                grp.create_dataset('frame_gps', data=frame_gps,
                                   dtype='float64', compression='gzip')
                grp.create_dataset('frame_valid', data=frame_valid.astype(np.uint8),
                                   compression='gzip')
                grp.attrs['calib_frame_id'] = calib_id

            logger.success("Propagation data saved to HDF5")
        finally:
            # Повторно відкриваємо базу у read-only режимі
            self.database._load_hot_data()