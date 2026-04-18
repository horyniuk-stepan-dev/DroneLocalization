# Code Quality Analysis — Drone Topometric Localization

> **Analysis date:** 2026-04-18  
> **Codebase:** ~5,900 lines of application code (excluding tests, scripts, config)

---

## 1. What Works Well

### 1.1 Strong Configuration Architecture

The single `AppConfig` Pydantic model in [`config/config.py`](file:///d:/My%20Projects/DroneLocalization/config/config.py) is an exemplary design. Every tunable parameter is validated at startup, has a clear default, and is accessed uniformly via `get_cfg(config, "dot.path", default)`. This eliminates scattered magic numbers across 30+ files.

### 1.2 Robust Geometry Pipeline

[`GeometryTransforms`](file:///d:/My%20Projects/DroneLocalization/src/geometry/transformations.py) implements a mature validation pipeline (L23–L113) that checks scale bounds, aspect ratio, shear, perspective components, and determinant sign before accepting any affine or homography matrix. The automatic fallback from Homography → Full Affine (L163–L171) when MAGSAC++ returns a degenerate matrix is a production-grade safeguard.

### 1.3 Thread-Safe Model Management

[`ModelManager`](file:///d:/My%20Projects/DroneLocalization/src/models/model_manager.py) handles multiple AI models with:
- Thread-safe lazy loading via `threading.Lock` (L74)
- LRU-based VRAM eviction (`_ensure_vram_available`, L151–L163)
- Model pinning to prevent eviction during tracking sessions (L129–L140)
- Platform-specific safety (Windows Triton check for `torch.compile`, L98–L127)

### 1.4 5-DoF Pose Graph with Sparse Optimization

The [`PoseGraphOptimizer`](file:///d:/My%20Projects/DroneLocalization/src/geometry/pose_graph_optimizer.py) correctly uses a 5-variable state (center_x, center_y, log_sx, log_sy, θ) which preserves anisotropic scale — rare and correct for nadir drone imagery. The sparse Jacobian (`_build_jac_sparsity`, L265–L294) fed to `scipy.optimize.least_squares(method="trf")` gives ~100× speedup over dense LM.

### 1.5 Multi-Backend Feature Matching

[`FeatureMatcher.match()`](file:///d:/My%20Projects/DroneLocalization/src/localization/matcher.py#L129-L149) dynamically routes by descriptor dimension:
- 128-dim → LightGlue (neural matcher)
- Any other → fast Numpy L2 with Lowe's ratio test + Mutual Nearest Neighbor

This adaptive dispatch allows the system to work with both XFeat (64-dim) and ALIKED (128-dim) features without code changes.

### 1.6 Comprehensive Telemetry

[`Telemetry`](file:///d:/My%20Projects/DroneLocalization/src/utils/telemetry.py) provides both context-manager and decorator APIs, tracks min/max/avg per stage, and auto-dumps on process exit. This is well-integrated across `DatabaseBuilder`, `FeatureMatcher`, and `Localizer`.

### 1.7 Defensive Logging

Almost every failure path includes structured context in log messages (L99–L103, L138–L141 in `project.py`; L232–L238 in `model_manager.py`). Error messages describe the condition, the relevant parameters, and suggest remediation. This significantly accelerates debugging in production.

---

## 2. Risks & Technical Debt

### 2.1 Test Coverage: 6% — Critical Risk

**Location:** `coverage.txt`, all `src/` modules

The overall statement coverage is **6%**. Out of 2,812 statements, 2,645 are untested. Critical modules with **0% coverage**:

| Module | Statements | Impact |
|---|---|---|
| `calibration_propagation_worker.py` | 222 | Core propagation logic — untested |
| `database_builder.py` | 210 | DB creation — untested |
| `localizer.py` | 194 | Localization pipeline — untested |
| `model_manager.py` | 145 | Model lifecycle — untested |
| `calibration_dialog.py` | 363 | GUI dialogs — untested |
| `database_loader.py` | 90 | Data access — untested |
| `tracking_worker.py` | 90 | Realtime loop — untested |
| `matcher.py` | 82 | Feature matching — untested |

Only `coordinates.py` (100%), `image_preprocessor.py` (100%), `image_utils.py` (97%), and `transformations.py` (81%) have meaningful coverage.

### 2.2 God-Class Anti-Pattern: `DatabaseBuilder.build_from_video()` — 430 Lines

**Location:** [`database_builder.py:50–482`](file:///d:/My%20Projects/DroneLocalization/src/database/database_builder.py#L50-L482)

This single method handles video decoding, masking, feature extraction, inter-frame homography, keyframe selection, HDF5 writing, LanceDB insertion, keypoint video rendering, and progress reporting. It contains 3 nested function definitions (`prefetch_frames`, `_flush_mask_batch`, `_process_single_frame`). Extracting these into dedicated classes or pipeline stages would improve testability and readability.

### 2.3 God-Class: `Localizer.localize_frame()` — 365 Lines

**Location:** [`localizer.py:69–434`](file:///d:/My%20Projects/DroneLocalization/src/localization/localizer.py#L69-L434)

This method handles rotation search, global descriptor extraction, feature extraction, candidate iteration, homography estimation, RMSE computation, FOV projection, and confidence calculation. The embedded diagnostic logging (L374–L400) alone is ~30 lines.

### 2.4 Mixin Explosion in MainWindow

**Location:** [`main_window.py:18`](file:///d:/My%20Projects/DroneLocalization/src/gui/main_window.py#L18)

```python
class MainWindow(CalibrationMixin, DatabaseMixin, TrackingMixin, PanoramaMixin, QMainWindow):
```

The four mixins total **~74 KB** of code (calibration_mixin: 31KB, database_mixin: 20KB, tracking_mixin: 12KB, panorama_mixin: 10KB). All share `self.*` state implicitly. This creates tight implicit coupling — any mixin can read/write any attribute on `self`, and bugs from name collisions are non-obvious. This is a maintenance risk as the system grows.

### 2.5 Duplicated Affine Decomposition

**Location:** Two separate `decompose_affine_5dof` functions exist:
1. [`affine_utils.py:48–58`](file:///d:/My%20Projects/DroneLocalization/src/geometry/affine_utils.py#L48-L58) — standalone utility
2. [`pose_graph_optimizer.py:369–374`](file:///d:/My%20Projects/DroneLocalization/src/geometry/pose_graph_optimizer.py#L369-L374) — module-level duplicate

Both are identical in logic. The `CalibrationPropagationWorker` imports from `affine_utils`, but `PoseGraphOptimizer` uses its own copy. This risks divergence when one is updated but not the other.

### 2.6 `_save_to_hdf5` Casts float64 to float32 Silently

**Location:** [`calibration_propagation_worker.py:438`](file:///d:/My%20Projects/DroneLocalization/src/workers/calibration_propagation_worker.py#L438)

```python
frame_affine[frame_id] = affine.astype(np.float32)
```

The entire pipeline (PoseGraphOptimizer, BFS, LM) operates in float64 precision. This cast to float32 at save-time introduces up to 1e-7 relative precision loss per matrix element. For large UTM coordinates (e.g., Easting 500,000 m), this truncation can introduce ~5 cm drift — particularly detrimental since the system applies a "Local Center" strategy specifically to mitigate this.

### 2.7 Hardcoded Failure Log Path

**Location:** [`localizer.py:581`](file:///d:/My%20Projects/DroneLocalization/src/localization/localizer.py#L581)

```python
csv_path = "logs/localization_failures.csv"
```

This path is relative to the CWD, not to the project directory. When the application is run from different directories (e.g., after PyInstaller packaging), the failure log goes to an unexpected location or silently fails.

### 2.8 No Graceful HDF5 Corruption Recovery

**Location:** [`database_loader.py:46–124`](file:///d:/My%20Projects/DroneLocalization/src/database/database_loader.py#L46-L124)

If the HDF5 file is truncated (e.g., power failure during `DatabaseBuilder` write), `h5py.File()` will throw an `OSError`. The `DatabaseLoader` logs it and re-raises, but there is no mechanism to detect partial databases, suggest truncation, or attempt repair. The SWMR mode enabled at L707 of `database_builder.py` helps, but a corrupted file still crashes the application at startup.

### 2.9 Global Mutable Singleton: `DEFAULT_CONVERTER`

**Location:** [`coordinates.py:128`](file:///d:/My%20Projects/DroneLocalization/src/geometry/coordinates.py#L128)

```python
DEFAULT_CONVERTER = CoordinateConverter("WEB_MERCATOR")
```

This module-level singleton is "for backward compatibility" but it's a global mutable state. If anything mutates it, all consumers are affected. There is no usage protection or freezing.

### 2.10 CESP Module: Dead Code Risk

**Location:** [`cesp_module.py`](file:///d:/My%20Projects/DroneLocalization/src/models/wrappers/cesp_module.py), [`model_manager.py:538–571`](file:///d:/My%20Projects/DroneLocalization/src/models/model_manager.py#L538-L571), [`config.py:99–103`](file:///d:/My%20Projects/DroneLocalization/config/config.py#L99-L103)

CESP (`CespConfig.enabled = False`) has a full code path (loader, config, model manager integration) but `weights_path` defaults to `None` and the feature is disabled. The warning "CESP initialized WITHOUT pretrained weights (random init)" at L557 means enabling it without trained weights will produce random noise in descriptors. No documentation explains how to train or obtain CESP weights.

### 2.11 GtsamPoseGraphOptimizer Is a No-Op

**Location:** [`pose_graph_optimizer.py:466–497`](file:///d:/My%20Projects/DroneLocalization/src/geometry/pose_graph_optimizer.py#L466-L497)

The `GtsamPoseGraphOptimizer` subclass always falls back to `super().optimize()` (the scipy TRF implementation). The GTSAM factor graph is initialized but never used. This adds an unused dependency path and misleading log messages ("GTSAM 5-DoF factor graph is currently mapped to SciPy TRF").

### 2.12 Mixed Language in Code Comments and UI Strings

Comments swap between Ukrainian and English unpredictably (e.g., `database_builder.py` has both `# Обчислюємо скільки кадрів РЕАЛЬНО буде оброблено` and `# Adaptive Keyframe Selection (П4)`). Error messages raised to the user are in Ukrainian (`"Не вдалося відкрити відео"`), while log messages are in English. For an open-source project, this creates an accessibility barrier.

---

## 3. Coupling Analysis

### 3.1 Tight Coupling: `Localizer` ↔ `DatabaseLoader` Internals

`Localizer._compute_confidence()` (L494–L537) directly accesses `self.database.frame_rmse[best_candidate_id]` and `self.database.frame_disagreement[best_candidate_id]` — bypassing the `DatabaseLoader` API. If the storage schema changes, this accessor breaks silently (returns `None` handled by ternary).

### 3.2 Circular Knowledge: `DatabaseBuilder` ↔ `FeatureMatcher`

[`DatabaseBuilder._compute_inter_frame_H()`](file:///d:/My%20Projects/DroneLocalization/src/database/database_builder.py#L553-L580) lazily creates a `FeatureMatcher` with `self._temp_model_manager` — a reference stashed as a side effect in `build_from_video()` (L60). This creates an implicit dependency that isn't visible in the constructor.

### 3.3 Config as God-Object

The `config` dict (or Pydantic model) is passed to nearly every class via constructor argument. While this is better than globals, many classes query config paths that belong to other layers (e.g., `DatabaseBuilder` reads `localization.fallback_extractor` at L178, and `CalibrationPropagationWorker` reads `localization.min_matches` at L62). This means any change to the config schema potentially affects unrelated modules.

---

## 4. Missing Abstractions

### 4.1 No Pipeline Abstraction

The system has a clear three-phase pipeline (Build → Calibrate → Localize) but no formalized pipeline or stage interface. Each phase is wired ad-hoc through mixins, workers, and signal/slot connections. A `Pipeline` abstraction with typed stage inputs/outputs would make the system easier to test, extend, and compose.

### 4.2 No Database Migration System

The HDF5 schema has at least 2 versions ("v1" and "v2") and the calibration format has versions "1.0", "2.0", "2.2", "3.0". Version detection is scattered across `DatabaseLoader.get_local_features()` (L246), `DatabaseLoader._load_propagation_data()` (L136–L162), and `MultiAnchorCalibration.load()` (L266–L292). There is no centralized migration or version compatibility matrix.

### 4.3 No Result Type / Error Type

`Localizer.localize_frame()` returns a raw `dict` with inconsistent shapes:
- On success: `{"success": True, "lat": ..., "lon": ..., "confidence": ..., "fov_polygon": ...}`
- On failure: `{"success": False, "error": str}` or `{"success": False, "error": str, "detail": str}`

A typed `LocalizationResult` dataclass (or `Result[T, E]` pattern) would eliminate key-existence checks throughout consumers.

### 4.4 No Feature Store Interface

`DatabaseBuilder` writes features, `DatabaseLoader` reads them, and `CalibrationPropagationWorker` accesses them via the reader. But there is no shared `FeatureStore` interface. Adding a new storage backend (e.g., SQLite, LMDB) would require modifying two classes simultaneously.
