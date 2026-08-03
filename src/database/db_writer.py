"""HDF5 + LanceDB persistence for the database build.

Extracted verbatim from ``DatabaseBuilder`` (IMPROVEMENT_PLAN п.1.3, розбиття
``db_builder``). Per the locked decomposition decision, ``DbWriter`` owns the
storage end-to-end: it creates the schema, holds the open ``h5py.File`` and the
LanceDB table, writes per-frame data, flushes the vector batch, builds the index
and closes everything. Nothing else in the build touches ``h5py`` or LanceDB.

Invariants pinned by ``tests/integration/test_db_builder_characterization.py``
and preserved here 1:1:

* ``write_pose`` is separate from ``save_frame_data`` — the caller writes a pose
  for EVERY processed slot, including non-keyframes (invariant 1), while full
  local features land only for keyframes (invariant 2).
* data is keyed by the processed slot index, and ``frame_step`` is persisted in
  metadata (invariant 3).
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from config import get_cfg
from src.utils.logging_utils import get_logger
from src.utils.telemetry import Telemetry

logger = get_logger(__name__)


class DbWriter:
    """Owns the HDF5 file and the LanceDB table for one build.

    ``descriptor_dim`` and ``local_descriptor_dim`` are attributes rather than
    constructor-only values because the builder detects both from live models
    after construction and before ``create_structure``.
    """

    def __init__(self, output_path, config: dict | None = None, descriptor_dim: int = 0):
        self.output_path = output_path
        self.config = config or {}
        self.descriptor_dim = descriptor_dim
        self.local_descriptor_dim = 128

        self.store_sift = get_cfg(self.config, "database.store_sift_features", False)
        self.sift_max_kps = get_cfg(self.config, "database.sift_max_keypoints", 2048)
        self.use_lancedb = get_cfg(self.config, "database.use_lancedb", True)
        self.lance_batch_size = get_cfg(self.config, "database.lancedb_batch_size", 64)
        self.lance_index_min_frames = get_cfg(self.config, "database.lancedb_index_min_frames", 256)

        self.db_file: h5py.File | None = None
        self.lance_table = None
        self.lance_batch: list = []

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_structure(
        self,
        num_frames: int,
        width: int,
        height: int,
        use_patchify: bool = False,
        num_patches: int = 0,
        frame_step: int = 1,
        source_total_frames: int = 0,
    ):
        """Create optimal HDF5 hierarchy with pre-allocated chunked arrays (schema v2)"""
        compression = get_cfg(self.config, "database.hdf5_compression", "lzf")
        chunk_f = get_cfg(self.config, "database.hdf5_chunk_frames", 64)
        max_kps = get_cfg(self.config, "database.max_keypoints_stored", 2048)
        local_desc_dim = self.local_descriptor_dim

        logger.info(
            f"Creating HDF5 v2 structure for {num_frames} frames "
            f"(compression={compression}, chunks={chunk_f}, max_kps={max_kps})"
        )

        if self.use_lancedb:
            # Imported lazily so the module (and its tests) load without the
            # lance/arrow stack when LanceDB is disabled.
            import lancedb
            import pyarrow as pa

            lance_path = Path(self.output_path).parent / "vectors.lance"
            if lance_path.exists():
                shutil.rmtree(lance_path)
            db = lancedb.connect(str(lance_path))
            schema = pa.schema(
                [
                    pa.field("frame_id", pa.int32()),
                    pa.field("vector", pa.list_(pa.float32(), self.descriptor_dim)),
                ]
            )
            self.lance_table = db.create_table("global_vectors", schema=schema, mode="create")
            self.lance_batch = []
            logger.info(f"LanceDB table created at {lance_path}")

        with h5py.File(self.output_path, "w", libver="latest") as f:
            # --- global_descriptors: chunked ---
            g1 = f.create_group("global_descriptors")
            if not self.use_lancedb:
                g1.create_dataset(
                    "descriptors",
                    shape=(num_frames, self.descriptor_dim),
                    maxshape=(None, self.descriptor_dim),
                    dtype="float32",
                    compression=compression,
                    chunks=(min(256, num_frames), self.descriptor_dim),
                )
            g1.create_dataset(
                "frame_poses",
                shape=(num_frames, 3, 3),
                maxshape=(None, 3, 3),
                dtype="float64",
                compression=compression,
                chunks=(min(256, num_frames), 3, 3),
            )

            # --- local_features: PRE-ALLOCATED chunked arrays (НОВА СХЕМА v2) ---
            lf = f.create_group("local_features")
            lf.create_dataset(
                "keypoints",
                shape=(num_frames, max_kps, 2),
                maxshape=(None, max_kps, 2),
                dtype="float32",
                compression=compression,
                chunks=(min(chunk_f, num_frames), max_kps, 2),
                fillvalue=0.0,
            )
            lf.create_dataset(
                "descriptors",
                shape=(num_frames, max_kps, local_desc_dim),
                maxshape=(None, max_kps, local_desc_dim),
                dtype="float16",  # float16: -50% розміру (П2)
                compression=compression,
                chunks=(min(chunk_f, num_frames), max_kps, local_desc_dim),
                fillvalue=0.0,
            )
            lf.create_dataset(
                "coords_2d",
                shape=(num_frames, max_kps, 2),
                maxshape=(None, max_kps, 2),
                dtype="float32",
                compression=compression,
                chunks=(min(chunk_f, num_frames), max_kps, 2),
                fillvalue=0.0,
            )
            lf.create_dataset(
                "kp_counts",  # скільки keypoints у кожному кадрі
                shape=(num_frames,),
                maxshape=(None,),
                dtype="int16",
                compression=compression,
                chunks=(min(num_frames, 4096),),
                fillvalue=0,
            )
            # Розміри кадру — зберігаємо ОДИН РАЗ у групі
            lf.attrs["frame_width"] = width
            lf.attrs["frame_height"] = height

            # --- RESEARCH 2.2: SIFT-ознаки для аварійного фолбека ---
            if self.store_sift:
                sf = f.create_group("sift_features")
                sf.create_dataset(
                    "keypoints",
                    shape=(num_frames, self.sift_max_kps, 2),
                    maxshape=(None, self.sift_max_kps, 2),
                    dtype="float32",
                    compression=compression,
                    chunks=(min(chunk_f, num_frames), self.sift_max_kps, 2),
                    fillvalue=0.0,
                )
                sf.create_dataset(
                    "descriptors",
                    shape=(num_frames, self.sift_max_kps, 128),
                    maxshape=(None, self.sift_max_kps, 128),
                    dtype="float16",  # rootSIFT ∈ [0,1] — f16 безпечний
                    compression=compression,
                    chunks=(min(chunk_f, num_frames), self.sift_max_kps, 128),
                    fillvalue=0.0,
                )
                sf.create_dataset(
                    "kp_counts",
                    shape=(num_frames,),
                    maxshape=(None,),
                    dtype="int16",
                    compression=compression,
                    chunks=(min(num_frames, 4096),),
                    fillvalue=0,
                )
                logger.info(f"SIFT fallback group created (max {self.sift_max_kps} kps/frame)")

            g3 = f.create_group("metadata")
            g3.attrs["num_frames"] = num_frames
            g3.attrs["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            g3.attrs["frame_width"] = width
            g3.attrs["frame_height"] = height
            g3.attrs["descriptor_dim"] = self.descriptor_dim
            g3.attrs["hdf5_schema"] = "v2"  # версія схеми для зворотної сумісності
            g3.attrs["max_keypoints"] = max_kps
            # Крок семплінгу відео: DB slot i = кадр відео i * frame_step.
            # Критично для калібрування — діалог конвертує номери кадрів відео у слоти БД.
            g3.attrs["frame_step"] = int(frame_step)
            g3.attrs["source_total_frames"] = int(source_total_frames)

            # Schema fingerprint: a stable hash of every structure/content-
            # defining setting (models, dims, keypoint budget, scale, frame
            # step, SIFT/VLAD policy). Lets databases built on different
            # machines be checked for interchangeability instead of silently
            # mixed. See src/database/schema_fingerprint.py. Never fatal.
            try:
                import json as _json

                from src.database.schema_fingerprint import (
                    build_components,
                    compute_fingerprint,
                )

                _fp_components = build_components(
                    self.config,
                    descriptor_dim=self.descriptor_dim,
                    local_descriptor_dim=self.local_descriptor_dim,
                    schema_version="v2",
                )
                g3.attrs["schema_fingerprint"] = compute_fingerprint(_fp_components)
                g3.attrs["schema_components"] = _json.dumps(_fp_components, sort_keys=True)
                logger.info(f"DB schema fingerprint: {g3.attrs['schema_fingerprint']}")
            except Exception as _fp_err:  # metadata must never break a build
                logger.warning(f"Could not write schema fingerprint: {_fp_err}")

            # Phase 2.2: Dataset for depth scales
            g3.create_dataset(
                "depth_scales",
                shape=(num_frames,),
                maxshape=(None,),
                dtype="float32",
                compression=compression,
                fillvalue=1.0,
            )

            # --- Patchify: мультимасштабні патч-дескриптори ---
            if use_patchify and num_patches > 0:
                pf = f.create_group("patch_descriptors")
                pf.create_dataset(
                    "descriptors",
                    shape=(num_frames, num_patches, self.descriptor_dim),
                    maxshape=(None, num_patches, self.descriptor_dim),
                    dtype="float32",
                    compression=compression,
                    chunks=(min(64, num_frames), num_patches, self.descriptor_dim),
                )
                g3.attrs["use_patchify"] = True
                g3.attrs["patchify_num_patches"] = num_patches
                logger.info(
                    f"Patchify HDF5 group created: {num_patches} patches × {self.descriptor_dim}D"
                )

            # Enable SWMR mode for parallel reading while writing
            f.swmr_mode = True

        logger.success("HDF5 v2 structure created successfully in SWMR mode")

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Reopens the created file for appending frame data."""
        self.db_file = h5py.File(self.output_path, "a")
        logger.info(f"Opened HDF5 file for writing: {self.output_path}")

    def write_pose(self, frame_id: int, pose_2d: np.ndarray) -> None:
        """Writes the pose for a slot regardless of keyframe status.

        ЗАВЖДИ зберігаємо pose для повного ланцюга пропагації, навіть якщо кадр
        не є keyframe (пропущений через малий рух). Без цього
        frame_poses[frame_id] = zeros → пропагація ламається.
        """
        if self.db_file:
            self.db_file["global_descriptors"]["frame_poses"][frame_id] = pose_2d

    def save_frame_data(self, frame_id: int, features: dict, pose_2d: np.ndarray):
        """Save extracted data for a single frame via slice assignment (schema v2)"""
        with Telemetry.profile("hdf5_write"):
            if self.use_lancedb:
                self.lance_batch.append({"frame_id": frame_id, "vector": features["global_desc"]})
                if len(self.lance_batch) >= self.lance_batch_size:
                    self.lance_table.add(self.lance_batch)
                    self.lance_batch = []
            else:
                self.db_file["global_descriptors"]["descriptors"][frame_id] = features[
                    "global_desc"
                ]

            self.db_file["global_descriptors"]["frame_poses"][frame_id] = pose_2d

            # local — slice assignment замість create_group + create_dataset
            kps = features["keypoints"]
            descs = features["descriptors"]
            c2d = features["coords_2d"]

            max_kps = self.db_file["local_features"]["keypoints"].shape[1]
            n = min(len(kps), max_kps)

            lf = self.db_file["local_features"]
            lf["keypoints"][frame_id, :n] = kps[:n]
            lf["descriptors"][frame_id, :n] = descs[:n].astype("float16")
            lf["coords_2d"][frame_id, :n] = c2d[:n]
            lf["kp_counts"][frame_id] = n

            # Patchify descriptors
            if "patch_descriptors" in features and "patch_descriptors" in self.db_file:
                self.db_file["patch_descriptors"]["descriptors"][frame_id] = features[
                    "patch_descriptors"
                ]

            # RESEARCH 2.2: SIFT-ознаки
            if "sift_keypoints" in features and "sift_features" in self.db_file:
                sf = self.db_file["sift_features"]
                s_kps = features["sift_keypoints"]
                s_descs = features["sift_descriptors"]
                sn = min(len(s_kps), sf["keypoints"].shape[1])
                if sn > 0:
                    sf["keypoints"][frame_id, :sn] = s_kps[:sn]
                    sf["descriptors"][frame_id, :sn] = s_descs[:sn].astype("float16")
                sf["kp_counts"][frame_id] = sn

            # Save depth scale
            if "depth_scale" in features:
                self.db_file["metadata"]["depth_scales"][frame_id] = features["depth_scale"]

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def finalize_vectors(self, saved_count: int) -> None:
        """Flushes the pending LanceDB batch and builds the IVF-PQ index."""
        if self.use_lancedb and self.lance_table is not None:
            if self.lance_batch:
                self.lance_table.add(self.lance_batch)
                self.lance_batch = []
            if saved_count >= self.lance_index_min_frames:
                logger.info("Building LanceDB IVF-PQ index...")
                self.lance_table.create_index(
                    metric="cosine",
                    num_partitions=min(256, saved_count // 8),
                    num_sub_vectors=32,
                )

    def write_frame_index_map(
        self,
        saved_count: int,
        frame_index_map: list,
        num_frames: int,
        use_keyframe_selection: bool,
    ) -> None:
        """Writes ``actual_num_frames`` / ``frame_index_map`` into metadata."""
        if self.db_file and saved_count > 0:
            try:
                meta = self.db_file["metadata"]
                meta.attrs["actual_num_frames"] = saved_count
                if "frame_index_map" not in meta:
                    meta.create_dataset(
                        "frame_index_map",
                        data=np.array(frame_index_map, dtype=np.int32),
                    )
                if use_keyframe_selection:
                    logger.info(
                        f"Keyframe selection: {saved_count}/{num_frames} frames saved "
                        f"({100 - saved_count / num_frames * 100:.1f}% reduction)"
                    )
            except Exception as e:
                logger.warning(f"Could not save frame_index_map: {e}")

    def close(self) -> None:
        if self.db_file:
            self.db_file.close()
            self.db_file = None
