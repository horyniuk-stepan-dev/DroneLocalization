import numpy as np
import h5py
import json
import math
from PyQt6.QtCore import QThread, pyqtSignal
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CalibrationPropagationWorker(QThread):
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

    def _decompose_affine(self, M):
        tx = float(M[0, 2])
        ty = float(M[1, 2])
        a = float(M[0, 0])
        b = float(M[0, 1])
        scale = math.sqrt(a ** 2 + b ** 2)
        angle = math.atan2(M[1, 0], M[0, 0])
        return tx, ty, scale, angle

    def _compose_affine(self, tx, ty, scale, angle):
        M = np.zeros((2, 3), dtype=np.float32)
        M[0, 0] = scale * math.cos(angle)
        M[0, 1] = -scale * math.sin(angle)
        M[1, 0] = scale * math.sin(angle)
        M[1, 1] = scale * math.cos(angle)
        M[0, 2] = tx
        M[1, 2] = ty
        return M

    def _propagate(self):
        num_frames = self.database.get_num_frames()
        anchors = sorted(self.calibration.anchors, key=lambda a: a.frame_id)

        if not anchors:
            self.error.emit("Немає якорів для розповсюдження")
            return

        self.progress.emit(10, "Розрахунок матриць (геометрична інтерполяція)...")
        frame_affine = np.zeros((num_frames, 2, 3), dtype=np.float32)
        frame_valid = np.ones(num_frames, dtype=bool)

        first_a = anchors[0]
        last_a = anchors[-1]

        for i in range(0, first_a.frame_id):
            frame_affine[i] = first_a.affine_matrix

        for i in range(last_a.frame_id, num_frames):
            frame_affine[i] = last_a.affine_matrix

        for i in range(len(anchors) - 1):
            a1 = anchors[i]
            a2 = anchors[i + 1]
            f1, f2 = a1.frame_id, a2.frame_id

            t1_x, t1_y, s1, ang1 = self._decompose_affine(a1.affine_matrix)
            t2_x, t2_y, s2, ang2 = self._decompose_affine(a2.affine_matrix)

            if ang2 - ang1 > math.pi:
                ang1 += 2 * math.pi
            elif ang1 - ang2 > math.pi:
                ang2 += 2 * math.pi

            for f in range(f1, f2 + 1):
                t = (f - f1) / float(f2 - f1) if f2 > f1 else 0
                t_x = (1 - t) * t1_x + t * t2_x
                t_y = (1 - t) * t1_y + t * t2_y
                s = (1 - t) * s1 + t * s2
                ang = (1 - t) * ang1 + t * ang2

                frame_affine[f] = self._compose_affine(t_x, t_y, s, ang)

        if not self._is_running:
            return

        self.progress.emit(50, "Збереження в базу даних HDF5...")
        db_path = self.database.db_path
        self.database.close()

        with h5py.File(db_path, 'a') as f:
            if 'calibration' in f:
                del f['calibration']
            grp = f.create_group('calibration')
            grp.create_dataset('frame_affine', data=frame_affine, dtype='float32', compression='gzip')
            grp.create_dataset('frame_valid', data=frame_valid.astype(np.uint8), compression='gzip')
            grp.attrs['anchors_json'] = json.dumps([a.to_dict() for a in anchors])

        self.database._load_hot_data()
        self.progress.emit(100, f"Готово! {num_frames} кадрів отримали ідеальні координати.")
        self.completed.emit()