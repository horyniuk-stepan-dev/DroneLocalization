# Database interchangeability & compute scaling

Goal: the program uses as much of each machine's compute as is available, **and**
every database it builds is structurally identical and interchangeable across
machines — the database format never depends on how powerful the machine is.

These two goals only coexist because config is split into two disjoint classes.

## 1. Two classes of config

**Performance keys (may scale with hardware).** Batch sizes, worker counts,
prefetch depth, thread counts, VRAM ratio, `torch.compile`, TensorRT, cuDNN/TF32.
These change *speed* only. They are auto-tuned at startup by
`src/utils/hardware_profile.py`. The complete allow-list is
`hardware_profile.TUNABLE_KEYS`:

- `models.vram_management.max_vram_ratio`
- `models.performance.torch_compile`
- `models.performance.fp16_enabled`
- `models.performance.propagation_max_workers`
- `database.yolo_batch_size`
- `database.prefetch_queue_size`
- `database.decode_batch_size`

`auto_tune()` is **fail-closed**: `_propose()` refuses any dot-path not in
`TUNABLE_KEYS` and logs an error. Adding a new auto-tuned key requires adding it
to that set on purpose.

**Structure / content keys (must NOT depend on hardware).** These define the
database's schema and content-type, so they are fixed in config and identical on
every machine:

- `models.local_extractor` (ALIKED vs RDD are not cross-matchable)
- `global_descriptor.backend` and its `descriptor_dim`
- `models.vlad.enabled`, `models.vlad.pca_dim` (change the global descriptor dim)
- `database.max_keypoints_stored` (HDF5 dataset shape)
- `database.keypoint_video_scale`
- `database.frame_step`
- `database.store_sift_features`, `database.sift_max_keypoints`

> Historical bug (fixed): `get_default_local_extractor()` used to pick the local
> extractor by VRAM (ALIKED under 8 GB, RDD otherwise). That made a weak machine
> build an ALIKED database and a strong one an RDD database — not interchangeable.
> It is now the fixed constant `CANONICAL_LOCAL_EXTRACTOR = "aliked"`. Change it
> only by editing config explicitly, and rebuild every database when you do.

## 2. Schema fingerprint

Every database records a fingerprint of all the structure/content keys in its
HDF5 metadata (`schema_fingerprint` + full `schema_components`), computed by
`src/database/schema_fingerprint.py` and written by `database_builder.py`.

`MultiDatabaseManager` checks the fingerprints of all loaded databases and logs a
loud error naming the exact differing field if any two disagree — so an
incompatible database is detected instead of silently corrupting matches. Two
databases with the same fingerprint are guaranteed interchangeable; the fingerprint
is deterministic and identical regardless of the machine that built them.

## 3. What "identical" means here

Structural interchangeability: same schema, dimensions, keypoint budget,
resolution scale and models on every machine. Descriptor *values* may differ by a
few floating-point ULPs between machines because FP16 autocast / TF32 / cuDNN
autotuning stay enabled for speed — that does not affect matching and does not
change the database structure. (If you ever need bit-identical descriptor values
across machines, lock FP32 and disable TF32/cuDNN-autotune/TensorRT for the
feature models; this costs some GPU throughput.)

## 4. Compute scaling — current state

Already scaled to the hardware (flexible):

- GPU used whenever available, automatic CPU fallback (`model_manager`).
- `apply_torch_backends()`: `torch`/OpenCV thread counts = physical cores;
  cuDNN benchmark on; TF32 matmul on Ampere+ GPUs.
- Auto-tuned per tier: decode batch, YOLO-mask micro-batch, prefetch depth,
  propagation workers, VRAM ratio, `torch.compile` (high/ultra), TensorRT-YOLO.
- `auto_tune` defaults to on; disable with `models.performance.auto_tune=false`.

Remaining upside (optional, needs on-hardware benchmarking — not enabled):

- **Batched feature extraction in the builder** (perf plan A5). The builder still
  runs the global/local feature models one frame at a time even though
  `FeatureExtractor.extract_features_batch` (CUDA streams) exists. Batching is the
  biggest single-GPU build-throughput win. Safe for interchangeability *iff*
  per-frame outputs are unchanged by batching — must be verified on a real GPU.
- **Multi-GPU.** `HardwareProfile` detects multiple GPUs but only `cuda:0` is
  used. Real multi-GPU dispatch (data-parallel build/localize) is a larger change.

Neither of these changes the database format, so enabling them later keeps every
existing database valid.
