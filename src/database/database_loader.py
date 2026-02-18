import h5py
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatabaseLoader:
    """Loads and manages access to the HDF5 topometric database"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db_file = None
        self.global_descriptors = None
        self.frame_poses = None
        self.metadata = {}

        # Дані пропагації калібрування (заповнюються після калібрування)
        self.h_to_calib = None      # (N, 3, 3) — H(frame_i → calib_frame)
        self.frame_gps = None       # (N, 2)    — (lat, lon) центру кожного кадру
        self.frame_valid = None     # (N,)      — True якщо кадр має GPS
        self.calib_frame_id = None  # int

        logger.info(f"Initializing DatabaseLoader with path: {db_path}")
        self._load_hot_data()

    def _load_hot_data(self):
        """Load global descriptors, poses and propagation data into RAM"""
        logger.info("Loading hot data into RAM...")

        try:
            self.db_file = h5py.File(self.db_path, 'r')

            self.global_descriptors = self.db_file['global_descriptors']['descriptors'][:]
            self.frame_poses = self.db_file['global_descriptors']['frame_poses'][:]

            logger.info(f"Loaded global descriptors: shape {self.global_descriptors.shape}")
            logger.info(f"Loaded frame poses: shape {self.frame_poses.shape}")

            for key, value in self.db_file['metadata'].attrs.items():
                self.metadata[key] = value
                logger.debug(f"Metadata - {key}: {value}")

            # Завантажуємо дані пропагації якщо є
            self._load_propagation_data()

            logger.success("Hot data loaded successfully into RAM")

        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            raise

    def _load_propagation_data(self):
        if 'calibration' not in self.db_file:
            logger.info("No propagation data in database (not calibrated yet)")
            self.frame_affine = None
            self.frame_valid = None
            return
        try:
            grp = self.db_file['calibration']
            if 'frame_affine' in grp:
                self.frame_affine = grp['frame_affine'][:]
                self.frame_valid = grp['frame_valid'][:].astype(bool)
                valid_count = int(np.sum(self.frame_valid))
                logger.success(f"Propagation data loaded: {valid_count} frames valid")
            else:
                logger.warning("Found old calibration format. Please run propagation again.")
                self.frame_affine = None
                self.frame_valid = None
        except Exception as e:
            logger.warning(f"Failed to load propagation data: {e}")
            self.frame_affine = None
            self.frame_valid = None

    @property
    def is_propagated(self) -> bool:
        return getattr(self, 'frame_affine', None) is not None

    def get_frame_affine(self, frame_id: int) -> np.ndarray | None:
        """Повертає унікальну афінну матрицю для конкретного кадру"""
        if not self.is_propagated:
            return None
        if frame_id < 0 or frame_id >= len(self.frame_valid):
            return None
        if not self.frame_valid[frame_id]:
            return None
        return self.frame_affine[frame_id]

    def get_h_to_anchor(self, frame_id: int):
        """
        Повертає (H_to_anchor, anchor_frame_id) для вказаного frame_id,
        або None якщо пропагація не виконана.

        Поточна реалізація: CalibrationPropagationWorker зберігає вже готову
        affine матрицю для кожного кадру напряму (через інтерполяцію між якорями),
        тому окремий H(ref→anchor) не потрібен — localizer читає frame_affine напряму
        через get_frame_affine(). Метод залишений для сумісності з API
        localizer._get_anchor_for_ref() і завжди повертає None,
        щоб localizer йшов основним шляхом через get_frame_affine().
        """
        return None

    def get_local_features(self, frame_id: int) -> dict:
        group_name = f'local_features/frame_{frame_id}'
        if group_name not in self.db_file:
            raise ValueError(f"Кадр {frame_id} не знайдено у базі даних.")
        g = self.db_file[group_name]
        return {
            'keypoints': g['keypoints'][:],
            'descriptors': g['descriptors'][:],
            'coords_2d': g['coords_2d'][:]
        }

    def get_num_frames(self) -> int:
        return int(self.metadata.get('num_frames', 0))

    def close(self):
        if self.db_file is not None:
            self.db_file.close()
            self.db_file = None
            logger.info("Database file closed")