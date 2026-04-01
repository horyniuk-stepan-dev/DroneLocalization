import json
from typing import Any

import h5py
import numpy as np

from src.geometry.coordinates import CoordinateConverter
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatabaseLoader:
    """Loads and manages access to the HDF5 topometric database (XFeat + DINOv2)"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db_file: h5py.File | None = None
        self.global_descriptors: np.ndarray | None = None
        self.frame_poses: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        self.converter: CoordinateConverter | None = None

        # Дані пропагації калібрування (заповнюються після калібрування)
        self.frame_affine: np.ndarray | None = None  # (N, 2, 3) — Metric Affine Matrices
        self.frame_valid: np.ndarray | None = None  # (N,)      — True якщо кадр має GPS
        self.frame_rmse: np.ndarray | None = None  # (N,)      — RMSE кожного кадру
        self.frame_disagreement: np.ndarray | None = None  # (N,)   — Розбіжність між гілками
        self.frame_matches: np.ndarray | None = None  # (N,)      — Кількість точок (inliers)

        # Каш для методів (заміна lru_cache для уникнення B019)
        self._size_cache: dict[int, tuple[int, int]] = {}
        self._feature_cache: dict[int, dict[str, np.ndarray]] = {}

        logger.info(f"Initializing DatabaseLoader | path={db_path}")
        self._load_hot_data()

    def _load_hot_data(self) -> None:
        """Load global descriptors (DINOv2), poses and propagation data into RAM"""
        logger.info(f"Loading hot data into RAM from: {self.db_path}")

        try:
            self.db_file = h5py.File(self.db_path, "r")
            logger.debug(f"HDF5 file opened | top-level groups: {list(self.db_file.keys())}")

            if "global_descriptors" not in self.db_file:
                raise KeyError(
                    f"HDF5 file is missing 'global_descriptors' group. "
                    f"Available groups: {list(self.db_file.keys())}. "
                    f"The database file may be corrupted or was created with an incompatible version."
                )

            self.global_descriptors = self.db_file["global_descriptors"]["descriptors"][:]
            self.frame_poses = self.db_file["global_descriptors"]["frame_poses"][:]

            logger.info(
                f"Loaded global descriptors: shape={self.global_descriptors.shape}, "
                f"dtype={self.global_descriptors.dtype}, "
                f"mem={self.global_descriptors.nbytes / 1024**2:.1f} MB"
            )
            logger.info(f"Loaded frame poses: shape={self.frame_poses.shape}")

            for key, value in self.db_file["metadata"].attrs.items():
                self.metadata[key] = value
                logger.debug(f"Metadata — {key}: {value}")

            if "actual_num_frames" in self.metadata:
                actual_num = int(self.metadata["actual_num_frames"])
                total_slots = len(self.global_descriptors)
                logger.info(
                    f"Database contains {actual_num} actual frames in {total_slots} pre-allocated slots"
                )
                # DO NOT SLICE with actual_num_frames! The arrays are sized to num_frames exactly,
                # and are indexed by absolute visual frame_id!

            if "frame_index_map" in self.db_file["metadata"]:
                self.frame_index_map = self.db_file["metadata"]["frame_index_map"][:]
                logger.debug(f"Frame index map loaded: {len(self.frame_index_map)} entries")
            else:
                self.frame_index_map = np.arange(len(self.global_descriptors))
                logger.debug("No frame_index_map found — using sequential indices")

            # Завантажуємо дані пропагації якщо є
            self._load_propagation_data()

            logger.success(
                f"Hot data loaded successfully | "
                f"{len(self.global_descriptors)} frames, "
                f"descriptor_dim={self.global_descriptors.shape[1]}, "
                f"propagated={'yes' if self.is_propagated else 'no'}"
            )

        except KeyError as e:
            logger.error(
                f"Database structure error: {e} | path={self.db_path}. "
                f"This usually means the HDF5 file is incomplete or was created with a different version."
            )
            raise
        except OSError as e:
            logger.error(
                f"Cannot open database file: {e} | path={self.db_path}. "
                f"Check that the file exists and is not locked by another process."
            )
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading database: {e} | path={self.db_path}", exc_info=True)
            raise

    def _load_propagation_data(self) -> None:
        if self.db_file is None or "calibration" not in self.db_file:
            logger.info("No propagation data in database (not calibrated yet)")
            self.frame_affine = None
            self.frame_valid = None
            return
        try:
            grp = self.db_file["calibration"]

            # 1. Відновлення проєкції (пріоритет)
            if "projection_json" in grp.attrs:
                try:
                    meta = json.loads(grp.attrs["projection_json"])
                    self.converter = CoordinateConverter.from_metadata(meta)
                    logger.success(f"Projection restored from HDF5: {meta['mode']}")
                except Exception as e:
                    logger.warning(
                        f"Could not load projection metadata: {e}. "
                        f"Raw value: {grp.attrs.get('projection_json', '<missing>')}. "
                        f"Falling back to default projection."
                    )
            elif "reference_gps" in grp.attrs:
                # Fallback для v2.0 (UTM)
                try:
                    ref_gps = json.loads(grp.attrs["reference_gps"])
                    self.converter = CoordinateConverter("UTM", tuple(ref_gps))
                    logger.success(f"UTM auto-initialized from legacy reference GPS: {ref_gps}")
                except Exception as e:
                    logger.warning(
                        f"Could not init UTM from legacy attribute: {e}. "
                        f"Raw reference_gps value: {grp.attrs.get('reference_gps', '<missing>')}. "
                        f"Defaulting to WEB_MERCATOR."
                    )
            else:
                # Fallback для v1.0 (WebMercator)
                logger.info("No projection metadata found. Defaulting to WEB_MERCATOR fallback.")
                self.converter = CoordinateConverter("WEB_MERCATOR")

            # 2. Завантаження датасетів
            if "frame_affine" in grp:
                self.frame_affine = grp["frame_affine"][:]
                self.frame_valid = grp["frame_valid"][:].astype(bool)

                # Метрики якості (QA)
                self.frame_rmse = grp["frame_rmse"][:] if "frame_rmse" in grp else None
                self.frame_disagreement = (
                    grp["frame_disagreement"][:] if "frame_disagreement" in grp else None
                )
                self.frame_matches = grp["frame_matches"][:] if "frame_matches" in grp else None

                valid_count = int(np.sum(self.frame_valid))
                logger.success(f"Propagation data loaded: {valid_count} frames valid")
            else:
                logger.warning("Found calibration group but no frame_affine dataset.")
                self.frame_affine = None
                self.frame_valid = None
        except Exception as e:
            logger.error(
                f"Failed to load propagation data: {e} | db_path={self.db_path}. "
                f"Calibration data may be corrupted — recalibration recommended.",
                exc_info=True,
            )
            self.frame_affine = None
            self.frame_valid = None

    @property
    def is_propagated(self) -> bool:
        return self.frame_affine is not None

    def get_frame_affine(self, frame_id: int) -> np.ndarray | None:
        """Повертає афінну матрицю для конкретного кадру"""
        if not self.is_propagated or self.frame_affine is None or self.frame_valid is None:
            return None
        if frame_id < 0 or frame_id >= len(self.frame_valid):
            return None
        if not self.frame_valid[frame_id]:
            return None
        return self.frame_affine[frame_id]

    def get_frame_size(self, frame_id: int) -> tuple[int, int]:
        """Повертає (height, width) для вказаного кадру"""
        if frame_id in self._size_cache:
            return self._size_cache[frame_id]

        if self.db_file is None:
            return 1080, 1920

        # Схема v2: розміри збережені один раз в local_features.attrs
        schema = self.metadata.get("hdf5_schema", "v1")
        if schema == "v2" and "local_features" in self.db_file:
            lf_attrs = self.db_file["local_features"].attrs
            h = int(lf_attrs.get("frame_height", self.metadata.get("frame_height", 1080)))
            w = int(lf_attrs.get("frame_width", self.metadata.get("frame_width", 1920)))
            self._size_cache[frame_id] = (h, w)
            return h, w

        # Схема v1: fallback — спочатку metadata, потім група кадру
        h = self.metadata.get("frame_height") or self.metadata.get("height") or 1080
        w = self.metadata.get("frame_width") or self.metadata.get("width") or 1920
        group_name = f"local_features/frame_{frame_id}"
        if group_name in self.db_file:
            g = self.db_file[group_name]
            if "height" in g.attrs and "width" in g.attrs:
                h, w = int(g.attrs["height"]), int(g.attrs["width"])

        res = (int(h), int(w))
        self._size_cache[frame_id] = res
        return res

    def get_local_features(self, frame_id: int) -> dict[str, np.ndarray]:
        """Повертає локальні ознаки XFeat для вказаного кадру"""
        if frame_id in self._feature_cache:
            return self._feature_cache[frame_id]

        if self.db_file is None:
            raise RuntimeError(
                f"Cannot get features for frame {frame_id}: database file is not opened. "
                f"Call _load_hot_data() first or check that db_path={self.db_path} exists."
            )

        schema = self.metadata.get("hdf5_schema", "v1")
        g = self.db_file["local_features"]

        if schema == "v2":
            n = int(g["kp_counts"][frame_id])
            if n == 0:
                raise ValueError(f"Кадр {frame_id} не має keypoints (kp_count=0).")
            res = {
                "keypoints": g["keypoints"][frame_id, :n],
                "descriptors": g["descriptors"][frame_id, :n].astype(np.float32),  # float16→32
                "coords_2d": g["coords_2d"][frame_id, :n],
            }
        else:
            # Стара схема v1 — зворотня сумісність
            if f"frame_{frame_id}" in g:
                old_g = g[f"frame_{frame_id}"]
                res = {
                    "keypoints": old_g["keypoints"][:],
                    "descriptors": old_g["descriptors"][:],
                    "coords_2d": old_g["coords_2d"][:],
                }
            else:
                # v1 pre-allocated без груп
                num = int(g["num_kp"][frame_id])
                if num == 0:
                    raise ValueError(f"Кадр {frame_id} не має keypoints (num_kp=0).")
                res = {
                    "keypoints": g["keypoints"][frame_id, :num],
                    "descriptors": g["descriptors"][frame_id, :num].astype(np.float32),
                    "coords_2d": g["coords_2d"][frame_id, :num],
                }

        # Обмежуємо розмір кешу (аналог lru_cache з maxsize=200)
        if len(self._feature_cache) > 200:
            self._feature_cache.pop(next(iter(self._feature_cache)))

        self._feature_cache[frame_id] = res
        return res

    def get_num_frames(self) -> int:
        """Повертає кількість pre-allocated слотів у БД (індексація за абсолютним frame_id)."""
        return int(self.metadata.get("num_frames", 0))

    def close(self) -> None:
        if self.db_file is not None:
            self.db_file.close()
            self.db_file = None
            logger.info("Database file closed")

        # Очищення кешу при закритті БД
        self._size_cache.clear()
        self._feature_cache.clear()