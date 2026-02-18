import numpy as np
import h5py
import json
from PyQt6.QtCore import QThread, pyqtSignal
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CalibrationPropagationWorker(QThread):
    """
    Миттєва пропагація GPS через лінійну інтерполяцію афінних матриць.
    Усуває проблему "дрейфу гомографії" та розриву ланцюжків кадрів.
    """

    progress = pyqtSignal(int, str)
    completed = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, database, calibration, matcher=None, config=None):
        super().__init__()
        self.database = database
        self.calibration = calibration
        self._is_running = True

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
            self.error.emit("Немає якорів для розповсюдження")
            return

        self.progress.emit(10, "Розрахунок матриць для всіх кадрів (Інтерполяція)...")

        # Масив для зберігання унікальної афінної матриці для КОЖНОГО кадру
        frame_affine = np.zeros((num_frames, 2, 3), dtype=np.float32)
        frame_valid = np.ones(num_frames, dtype=bool)

        first_a = anchors[0]
        last_a = anchors[-1]

        # 1. Заповнюємо хвости (до першого якоря і після останнього)
        for i in range(0, first_a.frame_id):
            frame_affine[i] = first_a.affine_matrix

        for i in range(last_a.frame_id, num_frames):
            frame_affine[i] = last_a.affine_matrix

        # 2. Плавна інтерполяція між якорями
        for i in range(len(anchors) - 1):
            a1 = anchors[i]
            a2 = anchors[i + 1]
            f1, f2 = a1.frame_id, a2.frame_id
            M1, M2 = a1.affine_matrix, a2.affine_matrix

            for f in range(f1, f2 + 1):
                t = (f - f1) / float(f2 - f1)
                # Лінійне змішування матриць
                frame_affine[f] = (1 - t) * M1 + t * M2

        if not self._is_running:
            return

        self.progress.emit(50, "Збереження в базу даних...")

        # 3. Зберігаємо результати у HDF5
        db_path = self.database.db_path
        self.database.close()

        with h5py.File(db_path, 'a') as f:
            if 'calibration' in f:
                del f['calibration']
            grp = f.create_group('calibration')
            grp.create_dataset('frame_affine', data=frame_affine, dtype='float32', compression='gzip')
            grp.create_dataset('frame_valid', data=frame_valid.astype(np.uint8), compression='gzip')

            anchors_json = json.dumps([a.to_dict() for a in anchors])
            grp.attrs['anchors_json'] = anchors_json

        # Перевідкриваємо базу для подальшої роботи
        self.database._load_hot_data()

        self.progress.emit(100, f"Готово! {num_frames} кадрів отримали ідеальні координати.")
        self.completed.emit()