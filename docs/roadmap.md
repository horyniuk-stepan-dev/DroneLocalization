# Improvement Roadmap — Drone Topometric Localization

> Sorted by **Impact × Effort** (highest impact, lowest effort first)

---

## 1. Fix float32 Truncation in Calibration Save

| | |
|---|---|
| **Problem** | `CalibrationPropagationWorker._save_to_hdf5()` casts optimized float64 affine matrices to float32 at [`calibration_propagation_worker.py:438`](file:///d:/My%20Projects/DroneLocalization/src/workers/calibration_propagation_worker.py#L438). For UTM Easting (e.g., 500,000 m), float32 has ~5 cm precision loss — undermining the Local Center strategy designed to prevent exactly this. |
| **Solution** | Keep float64 throughout the save path. |
| **Effort** | **S** |
| **Impact** | **High** |

```python
# calibration_propagation_worker.py:438
# Before:
frame_affine[frame_id] = affine.astype(np.float32)
# After:
frame_affine[frame_id] = affine  # already float64 from optimizer
```

---

## 2. Eliminate Duplicated `decompose_affine_5dof`

| | |
|---|---|
| **Problem** | Two identical `decompose_affine_5dof` functions exist: one in [`affine_utils.py:48`](file:///d:/My%20Projects/DroneLocalization/src/geometry/affine_utils.py#L48) and one in [`pose_graph_optimizer.py:369`](file:///d:/My%20Projects/DroneLocalization/src/geometry/pose_graph_optimizer.py#L369). If one is updated without the other, decomposition/composition symmetry breaks, causing silent coordinate drift. |
| **Solution** | Delete the duplicate from `pose_graph_optimizer.py` and import from `affine_utils`. |
| **Effort** | **S** |
| **Impact** | **High** |

```python
# pose_graph_optimizer.py — replace module-level duplicate
# Delete lines 369–374 and add import at the top:
from src.geometry.affine_utils import decompose_affine_5dof
```

---

## 3. Type the Localization Result

| | |
|---|---|
| **Problem** | `Localizer.localize_frame()` returns a bare `dict` with inconsistent keys on success vs failure. Consumers must do fragile `result.get("success")` / `result.get("error")` checks. IDE autocompletion and static analysis are impossible. |
| **Solution** | Introduce typed result dataclasses. |
| **Effort** | **S** |
| **Impact** | **Medium** |

```python
# src/localization/result.py (new file)
from dataclasses import dataclass

@dataclass(frozen=True)
class LocalizationSuccess:
    lat: float
    lon: float
    confidence: float
    matched_frame: int
    inliers: int
    fov_polygon: list[tuple[float, float]] | None
    sample_spread_m: float = 0.0
    fallback_mode: str | None = None
    is_of: bool = False

@dataclass(frozen=True)
class LocalizationFailure:
    error: str
    detail: str = ""

LocalizationResult = LocalizationSuccess | LocalizationFailure
```

---

## 4. Add Unit Tests for Core Domain Logic

| | |
|---|---|
| **Problem** | 0% coverage on `Localizer`, `FeatureMatcher`, `DatabaseBuilder`, `CalibrationPropagationWorker`, `KalmanFilter`, and `OutlierDetector`. Regressions in geometric math have no safety net. |
| **Solution** | Add focused unit tests for the pure-logic components that don't require GPU. Start with the components that have the highest blast radius. |
| **Effort** | **M** |
| **Impact** | **High** |

```python
# tests/test_outlier_detector.py (new)
import numpy as np
from src.tracking.outlier_detector import OutlierDetector

def test_accepts_normal_movement():
    det = OutlierDetector(window_size=5, threshold_std=3.0, max_speed_mps=100.0)
    for i in range(5):
        det.add_position((float(i), 0.0), dt=1.0)
    assert not det.is_outlier((5.0, 0.0), dt=1.0)

def test_rejects_teleportation():
    det = OutlierDetector(window_size=5, threshold_std=3.0, max_speed_mps=100.0)
    for i in range(5):
        det.add_position((float(i), 0.0), dt=1.0)
    assert det.is_outlier((999.0, 0.0), dt=1.0)

def test_auto_reset_after_consecutive():
    det = OutlierDetector(window_size=5, threshold_std=3.0, max_speed_mps=10.0, max_consecutive=3)
    for i in range(5):
        det.add_position((float(i), 0.0), dt=1.0)
    # Three consecutive outliers → auto-reset
    for _ in range(3):
        det.is_outlier((999.0, 0.0), dt=1.0)
    assert not det.is_outlier((999.0, 0.0), dt=1.0)  # accepted after reset
```

Priority test targets (by blast radius):
1. `OutlierDetector` — pure logic, no deps
2. `TrajectoryFilter` — pure Kalman math
3. `FeatureMatcher._fast_numpy_match()` — pure numpy
4. `PoseGraphOptimizer` — BFS + residuals math
5. `MultiAnchorCalibration` — PCHIP interpolation
6. `DatabaseLoader.get_local_features()` — mock HDF5

---

## 5. Extract `DatabaseBuilder.build_from_video()` Into Pipeline Stages

| | |
|---|---|
| **Problem** | `build_from_video()` is a 430-line monolithic method ([`database_builder.py:50–482`](file:///d:/My%20Projects/DroneLocalization/src/database/database_builder.py#L50-L482)) with 3 nested closures. It cannot be unit-tested, profiled per-stage, or extended without modifying the entire method. |
| **Solution** | Decompose into a pipeline with explicit stages: `VideoDecoder` → `FrameMasker` → `FeatureExtractor` → `KeyframeSelector` → `DatabaseWriter`. Each stage takes and returns a typed intermediate result. |
| **Effort** | **L** |
| **Impact** | **High** |

```python
# Sketch of decomposition:
class FramePacket:
    """Typed intermediate result flowing through pipeline stages."""
    frame_id: int
    frame_bgr: np.ndarray
    frame_rgb: np.ndarray
    static_mask: np.ndarray | None = None
    features: dict | None = None
    pose: np.ndarray | None = None
    is_keyframe: bool = True

class PipelineStage(Protocol):
    def process(self, packet: FramePacket) -> FramePacket: ...

class MaskingStage:
    """Batch-aware YOLO masking stage."""
    def __init__(self, masking_strategy, batch_size: int = 2): ...
    def process_batch(self, packets: list[FramePacket]) -> list[FramePacket]: ...

class FeatureExtractionStage:
    """Wraps FeatureExtractor with pose chain tracking."""
    def process(self, packet: FramePacket) -> FramePacket: ...

class KeyframeSelectionStage:
    """Applies motion threshold to decide is_keyframe."""
    def process(self, packet: FramePacket) -> FramePacket: ...
```

---

## 6. Replace Mixin Architecture in MainWindow

| | |
|---|---|
| **Problem** | `MainWindow` inherits from 4 mixins (74 KB total) that share implicit `self.*` state. Any mixin can read/write any attribute, causing hidden coupling and name-collision risks. |
| **Solution** | Replace mixins with a composition-based controller pattern. Each controller owns its state and communicates via typed signals. |
| **Effort** | **L** |
| **Impact** | **Medium** |

```python
# src/gui/controllers/tracking_controller.py (new)
class TrackingController(QObject):
    """Owns tracking state and workers, exposes signals to MainWindow."""
    location_found = pyqtSignal(float, float, float, int)
    
    def __init__(self, model_manager, config, parent=None):
        super().__init__(parent)
        self._worker = None
        self._model_manager = model_manager
        self._config = config
    
    def start_tracking(self, video_source: str, localizer):
        self._worker = RealtimeTrackingWorker(video_source, localizer, ...)
        self._worker.location_found.connect(self.location_found)
        self._worker.start()

# MainWindow becomes thin glue:
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tracking_ctl = TrackingController(self.model_manager, self.config)
        self.calibration_ctl = CalibrationController(...)
        self.database_ctl = DatabaseController(...)
```

---

## 7. Introduce Database Schema Migration

| | |
|---|---|
| **Problem** | HDF5 schema versions ("v1", "v2") and calibration versions ("1.0", "2.0", "2.2", "3.0") are handled via scattered `if`-branches in `DatabaseLoader` and `MultiAnchorCalibration`. Adding v3 requires editing multiple files. |
| **Solution** | Create a `SchemaMigrator` that detects the version and upgrades in-place. |
| **Effort** | **M** |
| **Impact** | **Medium** |

```python
# src/database/schema_migrator.py (new)
class SchemaMigrator:
    """Upgrades HDF5 databases from older schemas to current."""

    CURRENT_VERSION = "v2"

    @staticmethod
    def detect_version(db_path: str) -> str:
        with h5py.File(db_path, "r") as f:
            return f["metadata"].attrs.get("hdf5_schema", "v1")

    @staticmethod
    def migrate(db_path: str) -> None:
        version = SchemaMigrator.detect_version(db_path)
        if version == SchemaMigrator.CURRENT_VERSION:
            return

        if version == "v1":
            SchemaMigrator._migrate_v1_to_v2(db_path)
            logger.success(f"Migrated {db_path} from v1 → v2")

    @staticmethod
    def _migrate_v1_to_v2(db_path: str) -> None:
        """Convert per-frame groups to chunked arrays."""
        with h5py.File(db_path, "a") as f:
            # Read all frame_N groups, build arrays, write as v2 schema
            ...
```

---

## 8. Make Failure Log Path Project-Relative

| | |
|---|---|
| **Problem** | [`localizer.py:581`](file:///d:/My%20Projects/DroneLocalization/src/localization/localizer.py#L581) hardcodes `csv_path = "logs/localization_failures.csv"` relative to CWD. After PyInstaller packaging or running from a different directory, the log is silently lost or lands in an unexpected location. |
| **Solution** | Accept a project-aware log directory in the config or derive from the database path. |
| **Effort** | **S** |
| **Impact** | **Medium** |

```python
# localizer.py — accept log_dir via config
def __init__(self, database, feature_extractor, matcher, calibration, config=None):
    ...
    self._failure_log_dir = get_cfg(
        self.config, "localization.failure_log_dir",
        str(Path(database.db_path).parent / "logs")
    )

def _log_failure(self, error_type: str, ...):
    csv_path = os.path.join(self._failure_log_dir, "localization_failures.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    ...
```

---

## 9. Remove or Complete GTSAM Stub

| | |
|---|---|
| **Problem** | [`GtsamPoseGraphOptimizer`](file:///d:/My%20Projects/DroneLocalization/src/geometry/pose_graph_optimizer.py#L466-L497) always falls back to the parent class. It emits misleading log messages ("Initializing GTSAM nonlinear factor graph") then immediately calls `super().optimize()`. This confuses maintainers and adds a dead dependency path. |
| **Solution** | Either remove the class entirely (recommended) or gate it behind a feature flag and complete the Similarity2 implementation. |
| **Effort** | **S** |
| **Impact** | **Low** |

```python
# Option A: Remove entirely
# Delete lines 458–497 in pose_graph_optimizer.py
# Remove `import gtsam` try/except block (lines 458–463)

# Option B: Gate behind explicit feature flag
class GtsamPoseGraphOptimizer(PoseGraphOptimizer):
    def optimize(self, ...):
        if not GTSAM_AVAILABLE:
            return super().optimize(...)
        # TODO: Implement Similarity2 factor graph
        raise NotImplementedError("GTSAM Sim2 optimizer not yet implemented")
```

---

## 10. Remove `DEFAULT_CONVERTER` Global Singleton

| | |
|---|---|
| **Problem** | [`coordinates.py:128`](file:///d:/My%20Projects/DroneLocalization/src/geometry/coordinates.py#L128) exports `DEFAULT_CONVERTER = CoordinateConverter("WEB_MERCATOR")` as global mutable state "for backward compatibility." If any consumer mutates it, all other consumers are silently affected. |
| **Solution** | Grep usages — if none remain, delete it. If usages exist, replace with explicit construction at the call site. |
| **Effort** | **S** |
| **Impact** | **Low** |

```python
# Step 1: Find all usages
# grep -r "DEFAULT_CONVERTER" src/

# Step 2: If no usages, delete line 128:
# DEFAULT_CONVERTER = CoordinateConverter("WEB_MERCATOR")  # DELETE

# Step 3: If usages exist, replace each with:
converter = CoordinateConverter("WEB_MERCATOR")
```

---

## 11. Add LanceDB Fallback When Index Unavailable

| | |
|---|---|
| **Problem** | [`database_loader.py:57–66`](file:///d:/My%20Projects/DroneLocalization/src/database/database_loader.py#L57-L66) — if the `vectors.lance/` directory exists but is corrupted (e.g., partial write), `lancedb.connect()` may succeed but `open_table()` will throw. In this case, `global_descriptors` stays `None` and all retrieval fails silently. |
| **Solution** | Add try/except around LanceDB load with automatic fallback to HDF5 descriptors. |
| **Effort** | **S** |
| **Impact** | **Medium** |

```python
# database_loader.py:57–66
lance_path = Path(self.db_path).parent / "vectors.lance"
if lance_path.exists():
    try:
        db = lancedb.connect(str(lance_path))
        self.lance_table = db.open_table("global_vectors")
        logger.info(f"LanceDB loaded: {self.lance_table.count_rows()} vectors")
    except Exception as e:
        logger.warning(
            f"LanceDB corrupted or incompatible: {e}. "
            f"Falling back to HDF5 descriptors."
        )
        self.lance_table = None

if self.lance_table is None and "descriptors" in self.db_file["global_descriptors"]:
    self.global_descriptors = self.db_file["global_descriptors"]["descriptors"][:]
```

---

## 12. Standardize Code Language to English

| | |
|---|---|
| **Problem** | Comments and UI strings alternate between Ukrainian and English (e.g., `"Не вдалося відкрити відео"` in `database_builder.py:97`, `"Prefetch: {i}/{num_frames}"` in `calibration_propagation_worker.py:222`). This creates an accessibility barrier for external contributors and complicates grep-based debugging. |
| **Solution** | Migrate all code comments to English. Keep UI-facing strings in Ukrainian but externalize them into a `locale/` resource file for future i18n support. |
| **Effort** | **M** |
| **Impact** | **Low** |

```python
# Example migration for calibration_propagation_worker.py
# Before:
self.progress.emit(0, "Передзавантаження фіч у RAM...")
# After:
from src.locale import tr
self.progress.emit(0, tr("Prefetching features into RAM..."))

# src/locale/__init__.py (new)
_STRINGS = {
    "Prefetching features into RAM...": "Передзавантаження фіч у RAM...",
    ...
}
def tr(key: str) -> str:
    """Return localized string (Ukrainian by default)."""
    return _STRINGS.get(key, key)
```

---

## Summary Matrix

| # | Item | Effort | Impact | Priority Score |
|---|---|---|---|---|
| 1 | Fix float32 truncation in calibration save | S | High | ★★★★★ |
| 2 | Eliminate duplicated `decompose_affine_5dof` | S | High | ★★★★★ |
| 3 | Type the localization result | S | Medium | ★★★★ |
| 4 | Add unit tests for core domain logic | M | High | ★★★★ |
| 5 | Extract `build_from_video()` into pipeline stages | L | High | ★★★ |
| 6 | Replace mixin architecture in MainWindow | L | Medium | ★★★ |
| 7 | Introduce database schema migration | M | Medium | ★★★ |
| 8 | Make failure log path project-relative | S | Medium | ★★★★ |
| 9 | Remove or complete GTSAM stub | S | Low | ★★★ |
| 10 | Remove `DEFAULT_CONVERTER` global singleton | S | Low | ★★★ |
| 11 | Add LanceDB fallback when index unavailable | S | Medium | ★★★★ |
| 12 | Standardize code language to English | M | Low | ★★ |
