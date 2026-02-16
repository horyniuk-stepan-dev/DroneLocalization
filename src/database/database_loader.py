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

        logger.info(f"Initializing DatabaseLoader with path: {db_path}")
        self._load_hot_data()

    def _load_hot_data(self):
        """Load global descriptors and poses into RAM for instant access"""
        logger.info("Loading hot data (global descriptors and poses) into RAM...")

        try:
            self.db_file = h5py.File(self.db_path, 'r')

            self.global_descriptors = self.db_file['global_descriptors']['descriptors'][:]
            self.frame_poses = self.db_file['global_descriptors']['frame_poses'][:]

            logger.info(f"Loaded global descriptors: shape {self.global_descriptors.shape}")
            logger.info(f"Loaded frame poses: shape {self.frame_poses.shape}")

            for key, value in self.db_file['metadata'].attrs.items():
                self.metadata[key] = value
                logger.debug(f"Metadata - {key}: {value}")

            logger.success("Hot data loaded successfully into RAM")

        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            raise

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
