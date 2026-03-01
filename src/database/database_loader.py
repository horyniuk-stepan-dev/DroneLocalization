import h5py
import numpy as np
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatabaseLoader:
    """
    Loads and manages access to the HDF5 topometric database.

    Hot data (global descriptors, poses, propagation affines) is loaded into RAM on init.
    Local features are read lazily from disk with an in-memory LRU-style cache.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db_file: h5py.File | None = None

        # Hot data — loaded into RAM
        self.global_descriptors: np.ndarray | None = None
        self.frame_poses: np.ndarray | None = None
        self.metadata: dict = {}

        # Propagation data — filled by CalibrationPropagationWorker
        self.frame_affine: np.ndarray | None = None  # (N, 2, 3) affine per frame
        self.frame_valid:  np.ndarray | None = None  # (N,) bool

        # Local features LRU cache
        self._local_features_cache: dict[int, dict] = {}
        self._cache_max_size = 256

        logger.info(f"DatabaseLoader: {db_path}")
        self._load_hot_data()

    # ── Context manager support

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()

    # ── Loading

    def _load_hot_data(self) -> None:
        try:
            self.db_file = h5py.File(self.db_path, 'r')

            self.global_descriptors = self.db_file['global_descriptors']['descriptors'][:]
            self.frame_poses        = self.db_file['global_descriptors']['frame_poses'][:]

            for key, value in self.db_file['metadata'].attrs.items():
                self.metadata[key] = value

            self._load_propagation_data()

            logger.success(
                f"Database loaded | "
                f"frames={self.global_descriptors.shape[0]}, "
                f"desc_dim={self.global_descriptors.shape[1]}, "
                f"propagated={self.is_propagated}"
            )
        except Exception as e:
            logger.error(f"Failed to load database: {e}", exc_info=True)
            raise

    def _load_propagation_data(self) -> None:
        if 'calibration' not in self.db_file:
            logger.info("No propagation data found — run calibration first")
            return

        try:
            grp = self.db_file['calibration']
            if 'frame_affine' not in grp:
                logger.warning("Old calibration format detected — please re-run propagation")
                return

            self.frame_affine = grp['frame_affine'][:]
            self.frame_valid  = grp['frame_valid'][:].astype(bool)
            logger.success(f"Propagation data loaded: {int(np.sum(self.frame_valid))} valid frames")

        except Exception as e:
            logger.warning(f"Failed to load propagation data: {e}")
            self.frame_affine = None
            self.frame_valid  = None

    # ── Properties

    @property
    def is_propagated(self) -> bool:
        return self.frame_affine is not None

    def get_num_frames(self) -> int:
        return int(self.metadata.get('num_frames', 0))

    # ── Data access

    def get_frame_affine(self, frame_id: int) -> np.ndarray | None:
        """Return pre-computed affine matrix for frame_id, or None if unavailable."""
        if not self.is_propagated:
            return None
        if not (0 <= frame_id < len(self.frame_valid)):
            return None
        if not self.frame_valid[frame_id]:
            return None
        return self.frame_affine[frame_id]

    def get_local_features(self, frame_id: int) -> dict:
        """
        Return local features for frame_id.
        Cached in RAM (up to _cache_max_size entries) to avoid repeated HDF5 reads
        for frequently-retrieved candidate frames.
        """
        if frame_id in self._local_features_cache:
            return self._local_features_cache[frame_id]

        group_name = f'local_features/frame_{frame_id}'
        if group_name not in self.db_file:
            raise ValueError(f"Frame {frame_id} not found in database")

        g = self.db_file[group_name]
        features = {
            'keypoints':   g['keypoints'][:],
            'descriptors': g['descriptors'][:],
            # backward-compat: coords_2d removed in new builder, fall back to keypoints
            'coords_2d':   g['coords_2d'][:] if 'coords_2d' in g else g['keypoints'][:],
        }

        # Simple FIFO eviction when cache is full
        if len(self._local_features_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._local_features_cache))
            del self._local_features_cache[oldest_key]

        self._local_features_cache[frame_id] = features
        return features

    # Lifecycle

    def close(self) -> None:
        if self.db_file is not None:
            try:
                self.db_file.close()
            except Exception:
                pass
            self.db_file = None
            self._local_features_cache.clear()
            logger.info("Database closed")
