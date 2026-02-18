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
        """Завантажити результати GPS-пропагації якщо вони збережені у HDF5"""
        if 'calibration' not in self.db_file:
            logger.info("No propagation data found in database (not calibrated yet)")
            return

        try:
            grp = self.db_file['calibration']
            self.h_to_calib = grp['h_to_calib'][:]
            self.frame_gps = grp['frame_gps'][:]
            self.frame_valid = grp['frame_valid'][:].astype(bool)
            self.calib_frame_id = int(grp.attrs['calib_frame_id'])

            valid_count = int(np.sum(self.frame_valid))
            logger.success(
                f"Propagation data loaded: {valid_count}/{len(self.frame_valid)} frames "
                f"have GPS, calib_frame_id={self.calib_frame_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to load propagation data: {e}")
            self.h_to_calib = None
            self.frame_gps = None
            self.frame_valid = None
            self.calib_frame_id = None

    @property
    def is_propagated(self) -> bool:
        """Чи є дані пропагації GPS для всіх кадрів"""
        return (self.h_to_calib is not None and
                self.frame_gps is not None and
                self.frame_valid is not None)

    def get_h_to_calib(self, frame_id: int) -> np.ndarray | None:
        """
        Отримати H(frame_id → calib_frame).
        Повертає None якщо пропагація не виконана або кадр невалідний.
        """
        if not self.is_propagated:
            return None
        if frame_id < 0 or frame_id >= len(self.frame_valid):
            return None
        if not self.frame_valid[frame_id]:
            return None
        return self.h_to_calib[frame_id]

    def get_frame_gps(self, frame_id: int) -> tuple | None:
        """
        Отримати попередньо обчислені GPS координати центру кадру.
        Повертає (lat, lon) або None якщо кадр невалідний.
        """
        if not self.is_propagated:
            return None
        if frame_id < 0 or frame_id >= len(self.frame_valid):
            return None
        if not self.frame_valid[frame_id]:
            return None
        lat, lon = self.frame_gps[frame_id]
        return float(lat), float(lon)

    def get_local_features(self, frame_id: int) -> dict:
        """Lazy load local features for a specific frame from disk"""
        logger.debug(f"Lazy loading local features for frame {frame_id}")

        group_name = f'local_features/frame_{frame_id}'
        if group_name not in self.db_file:
            logger.error(f"Frame {frame_id} not found in database")
            raise ValueError(f"Кадр {frame_id} не знайдено у базі даних.")

        frame_group = self.db_file[group_name]
        features = {
            'keypoints': frame_group['keypoints'][:],
            'descriptors': frame_group['descriptors'][:],
            'coords_2d': frame_group['coords_2d'][:]
        }

        logger.debug(f"Frame {frame_id}: Loaded {len(features['keypoints'])} keypoints")
        return features

    def get_num_frames(self) -> int:
        num_frames = self.metadata.get('num_frames', 0)
        logger.debug(f"Database contains {num_frames} frames")
        return num_frames

    def close(self):
        """Safely close the HDF5 file handle"""
        if self.db_file is not None:
            logger.info("Closing database file")
            self.db_file.close()
            self.db_file = None
            logger.success("Database file closed successfully")