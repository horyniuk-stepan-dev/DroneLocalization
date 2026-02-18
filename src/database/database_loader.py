import json
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

        # Multi-anchor propagation data
        self.h_to_anchor = None           # (N, 3, 3) H(frame_i → nearest anchor)
        self.nearest_anchor_fid = None    # (N,) frame_id якоря для кожного кадру
        self.frame_gps = None             # (N, 2) (lat, lon) для кожного кадру
        self.frame_valid = None           # (N,) bool

        logger.info(f"Initializing DatabaseLoader: {db_path}")
        self._load_hot_data()

    def _load_hot_data(self):
        logger.info("Loading hot data into RAM...")
        try:
            self.db_file = h5py.File(self.db_path, 'r')
            self.global_descriptors = self.db_file['global_descriptors']['descriptors'][:]
            self.frame_poses = self.db_file['global_descriptors']['frame_poses'][:]

            for key, value in self.db_file['metadata'].attrs.items():
                self.metadata[key] = value

            logger.info(f"Descriptors: {self.global_descriptors.shape}, "
                        f"Poses: {self.frame_poses.shape}")

            self._load_propagation_data()
            logger.success("Hot data loaded")
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            raise

    def _load_propagation_data(self):
        if 'calibration' not in self.db_file:
            logger.info("No propagation data in database (not calibrated yet)")
            return
        try:
            grp = self.db_file['calibration']
            self.h_to_anchor = grp['h_to_anchor'][:]
            self.nearest_anchor_fid = grp['nearest_anchor_frame_id'][:]
            self.frame_gps = grp['frame_gps'][:]
            self.frame_valid = grp['frame_valid'][:].astype(bool)

            valid_count = int(np.sum(self.frame_valid))
            num_anchors = int(grp.attrs.get('num_anchors', 1))
            logger.success(
                f"Propagation data loaded: {valid_count}/{len(self.frame_valid)} frames, "
                f"{num_anchors} anchors"
            )
        except Exception as e:
            logger.warning(f"Failed to load propagation data: {e}")
            self.h_to_anchor = None
            self.nearest_anchor_fid = None
            self.frame_gps = None
            self.frame_valid = None

    @property
    def is_propagated(self) -> bool:
        return (self.h_to_anchor is not None and
                self.frame_gps is not None and
                self.frame_valid is not None)

    def get_h_to_anchor(self, frame_id: int) -> tuple | None:
        """
        Повертає (H_to_anchor, anchor_frame_id) або None.
        H_to_anchor — це H(frame_id → nearest_anchor).
        """
        if not self.is_propagated:
            return None
        if frame_id < 0 or frame_id >= len(self.frame_valid):
            return None
        if not self.frame_valid[frame_id]:
            return None
        return self.h_to_anchor[frame_id], int(self.nearest_anchor_fid[frame_id])

    def get_frame_gps(self, frame_id: int) -> tuple | None:
        """Попередньо обчислені GPS координати центру кадру (lat, lon) або None"""
        if not self.is_propagated:
            return None
        if frame_id < 0 or frame_id >= len(self.frame_valid):
            return None
        if not self.frame_valid[frame_id]:
            return None
        return float(self.frame_gps[frame_id][0]), float(self.frame_gps[frame_id][1])

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