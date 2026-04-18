# Architecture — Drone Topometric Localization System

> **Version:** 1.0.0  
> **Python:** 3.11 · **PyTorch:** 2.2+ · **GUI:** PyQt6  
> **Last updated:** 2026-04-18

---

## 1. Module Map

```
DroneLocalization/
├── main.py                          — Application entry point, Qt bootstrap, prewarm
├── config/
│   └── config.py                    — Pydantic-validated AppConfig (single source of truth)
├── src/
│   ├── core/                        — Project lifecycle management
│   │   ├── project.py               — ProjectManager & ProjectSettings (dataclass)
│   │   ├── project_registry.py      — JSON-based cross-session project registry (~/.drone_localizer/)
│   │   └── export_results.py        — ResultExporter: CSV, GeoJSON, KML output
│   │
│   ├── models/                      — AI model loading, VRAM management, TRT integration
│   │   ├── model_manager.py         — ModelManager: thread-safe lazy loading, VRAM eviction, pinning
│   │   └── wrappers/
│   │       ├── feature_extractor.py — FeatureExtractor: DINOv2 global + XFeat/ALIKED local
│   │       ├── aliked_wrapper.py    — ALIKED keypoint adapter for LightGlue
│   │       ├── yolo_wrapper.py      — YOLOWrapper: instance segmentation → dynamic object masks
│   │       ├── masking_strategy.py  — Strategy pattern for YOLO / none masking
│   │       ├── cesp_module.py       — CESP multi-scale descriptor enhancement (optional)
│   │       └── trt_dinov2_wrapper.py— TensorRT FP16 wrapper for DINOv2 inference
│   │
│   ├── database/                    — Reference database build & load
│   │   ├── database_builder.py      — DatabaseBuilder: video → HDF5 v2 + LanceDB
│   │   └── database_loader.py       — DatabaseLoader: HDF5/LanceDB reader with LRU cache
│   │
│   ├── localization/                — Core localization pipeline
│   │   ├── localizer.py             — Localizer: DINOv2 retrieval → feature matching → Homography → GPS
│   │   └── matcher.py               — FeatureMatcher (LightGlue / Numpy L2), FastRetrieval, LanceDBRetrieval
│   │
│   ├── geometry/                    — Mathematical transforms & graph optimization
│   │   ├── transformations.py       — GeometryTransforms: Homography/Affine via OpenCV MAGSAC++ or PoseLib
│   │   ├── coordinates.py           — CoordinateConverter: WGS84 ↔ Web Mercator / UTM (via pyproj)
│   │   ├── affine_utils.py          — 4-DoF / 5-DoF affine decomposition & composition
│   │   └── pose_graph_optimizer.py  — PoseGraphOptimizer: 5-DoF LM via scipy.optimize (sparse TRF)
│   │
│   ├── calibration/                 — GPS anchor management & coordinate propagation
│   │   └── multi_anchor_calibration.py — MultiAnchorCalibration: PCHIP interpolation over anchor affines
│   │
│   ├── tracking/                    — Trajectory smoothing & anomaly detection
│   │   ├── kalman_filter.py         — TrajectoryFilter: 4-state Kalman (x, y, vx, vy)
│   │   └── outlier_detector.py      — OutlierDetector: speed-based Z-score with auto-reset
│   │
│   ├── workers/                     — QThread background tasks
│   │   ├── calibration_propagation_worker.py — Graph-based calibration propagation (5 phases)
│   │   ├── database_worker.py       — Async database build wrapper
│   │   ├── tracking_worker.py       — RealtimeTrackingWorker: keyframe + Optical Flow pipeline
│   │   ├── panorama_worker.py       — Video → stitched panorama
│   │   ├── panorama_overlay_worker.py — Panorama → georeferenced map overlay
│   │   └── video_decode_worker.py   — Decord/OpenCV video frame producer
│   │
│   ├── gui/                         — PyQt6 desktop interface
│   │   ├── main_window.py           — MainWindow with mixin-based feature composition
│   │   ├── mixins/                  — CalibrationMixin, DatabaseMixin, TrackingMixin, PanoramaMixin
│   │   ├── widgets/
│   │   │   ├── control_panel.py     — Left dock: all action buttons and settings
│   │   │   ├── video_widget.py      — Central: OpenCV frame display
│   │   │   └── map_widget.py        — Right dock: Leaflet map (PyQt6-WebEngine)
│   │   └── dialogs/
│   │       ├── calibration_dialog.py— Multi-point GPS anchor editor
│   │       ├── new_mission_dialog.py— Mission creation wizard
│   │       └── open_project_dialog.py— Recent projects browser
│   │
│   └── utils/                       — Shared utilities
│       ├── logging_utils.py         — Loguru wrapper (get_logger, setup_logging, silent_output)
│       ├── image_preprocessor.py    — CLAHE contrast enhancement
│       ├── image_utils.py           — Resize, crop, format conversion helpers
│       └── telemetry.py             — Runtime profiling (context-manager / decorator API)
│
├── tests/                           — 29 tests: unit + integration + benchmarks
├── scripts/                         — Build, export, migration, and debug tooling
└── models/                          — YOLO .pt/.engine/.onnx model artifacts
```

---

## 2. Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PRESENTATION LAYER                            │
│                                                                             │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐ │
│   │  ControlPanel    │  │  VideoWidget      │  │  MapWidget (Leaflet)     │ │
│   │  (Left Dock)     │  │  (Center)         │  │  (Right Dock)            │ │
│   └────────┬─────────┘  └────────┬──────────┘  └───────────┬─────────────┘ │
│            │                     │                         │               │
│   ┌────────┴─────────────────────┴─────────────────────────┴─────────────┐ │
│   │          MainWindow  (CalibrationMixin + DatabaseMixin +             │ │
│   │                       TrackingMixin + PanoramaMixin)                  │ │
│   └─────────────────────────────┬────────────────────────────────────────┘ │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │ signals / slots
┌─────────────────────────────────┼───────────────────────────────────────────┐
│                          WORKER LAYER (QThread)                             │
│                                                                             │
│   ┌───────────────┐ ┌──────────────────────┐ ┌────────────────────────────┐│
│   │ DatabaseWorker│ │RealtimeTrackingWorker│ │CalibrationPropagationWkr  ││
│   │(build_from_   │ │(keyframe + OF loop)  │ │(graph build → LM optimize)││
│   │  video)       │ │                      │ │                            ││
│   └───────┬───────┘ └──────────┬───────────┘ └────────────┬───────────────┘│
│           │                    │                          │                │
│   ┌───────┴────┐  ┌───────────┴──────────┐  ┌────────────┴──────────────┐ │
│   │Panorama    │  │ Video Decode Worker   │  │ PanoramaOverlayWorker    │ │
│   │Worker      │  │ (Decord / cv2)       │  │ (georeference & tile)    │ │
│   └────────────┘  └──────────────────────┘  └───────────────────────────┘ │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────┼───────────────────────────────────────────┐
│                          DOMAIN LAYER (Core Logic)                          │
│                                                                             │
│   ┌────────────────────┐    ┌───────────────────┐   ┌────────────────────┐ │
│   │   Localizer        │───▶│  FeatureMatcher   │   │ MultiAnchor        │ │
│   │ (orchestrator)     │    │ (LightGlue/L2)    │   │ Calibration        │ │
│   └──┬────────┬────────┘    └───────────────────┘   │ (PCHIP interp)    │ │
│      │        │                                     └────────────────────┘ │
│      │   ┌────┴───────────────────┐   ┌──────────────────────────────────┐ │
│      │   │  TrajectoryFilter      │   │  PoseGraphOptimizer              │ │
│      │   │  (Kalman 4-state)      │   │  (5-DoF LM + BFS init + GeoJSON)│ │
│      │   │  OutlierDetector       │   │                                  │ │
│      │   │  (Z-score + speed)     │   │  GtsamPoseGraphOptimizer (stub)  │ │
│      │   └────────────────────────┘   └──────────────────────────────────┘ │
│      │                                                                     │
│   ┌──┴──────────────────────────────────────────────────────────────┐      │
│   │                   GeometryTransforms                            │      │
│   │   estimate_homography (MAGSAC++ / PoseLib LO-RANSAC)           │      │
│   │   estimate_affine / estimate_affine_partial                    │      │
│   │   apply_homography / apply_affine                              │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   ┌──────────────────────────┐   ┌──────────────────────────────────┐      │
│   │   CoordinateConverter    │   │  ResultExporter (CSV/GeoJSON/KML)│      │
│   │   WGS84 ↔ Metric        │   └──────────────────────────────────┘      │
│   │   (pyproj)               │                                             │
│   └──────────────────────────┘                                             │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────┼───────────────────────────────────────────┐
│                       INFRASTRUCTURE LAYER                                  │
│                                                                             │
│   ┌──────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐ │
│   │  ModelManager     │  │  DatabaseBuilder     │  │  DatabaseLoader      │ │
│   │  (VRAM, pin,     │  │  (HDF5 v2 + LanceDB) │  │  (HDF5 reader,      │ │
│   │   eviction,      │  │                      │  │   LRU feature cache) │ │
│   │   TRT fallback)  │  │                      │  │                      │ │
│   └──────┬───────────┘  └──────────────────────┘  └──────────────────────┘ │
│          │                                                                  │
│   ┌──────┴───────────────────────────────────────────────────────────────┐  │
│   │                     Model Wrappers                                   │  │
│   │  ┌─────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ ┌─────┐ │  │
│   │  │DINOv2   │ │XFeat   │ │ALIKED    │ │LightGlue │ │YOLO  │ │CESP │ │  │
│   │  │(Global) │ │(Local) │ │(Local)   │ │(Matcher) │ │(Mask)│ │(Enh)│ │  │
│   │  └─────────┘ └────────┘ └──────────┘ └──────────┘ └──────┘ └─────┘ │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │  Storage:  HDF5 (hierarchical features)  ·  LanceDB (ANN search) │    │
│   │            JSON (project + calibration)  ·  GeoJSON (diagnostics) │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────────┐ │
│   │  Loguru (logging)  │  │  Telemetry         │  │  Pydantic (config)   │ │
│   │                    │  │  (perf_counter)     │  │                      │ │
│   └────────────────────┘  └────────────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow: End-to-End Pipeline

### 3.1 Database Creation

```
Reference Video (.mp4)
        │
        ▼
┌─────────────────────────────┐
│  DatabaseBuilder.            │
│  build_from_video()          │
│                              │
│  ┌────────────────────────┐  │         ┌──────────────────────────┐
│  │ Decord / cv2 decode    │──┼────────▶│ MaskingStrategy (YOLO)   │
│  │ (prefetch queue, 32)   │  │         │ get_mask_batch()         │
│  └──────────┬─────────────┘  │         └──────────┬───────────────┘
│             │ frame (RGB)    │                     │ static_mask
│             ▼                │                     ▼
│  ┌────────────────────────┐  │  ┌──────────────────────────────────┐
│  │ FeatureExtractor       │  │  │ Per-frame processing:            │
│  │  ├─ DINOv2 ──▶ global  │──┼─▶│  1. global_desc → LanceDB batch │
│  │  └─ ALIKED  ──▶ local  │  │  │  2. keypoints/desc → HDF5 slice │
│  └────────────────────────┘  │  │  3. inter-frame H → pose chain  │
│                              │  │  4. keyframe selection (motion)  │
│                              │  └──────────────────────────────────┘
└──────────────────────────────┘
        │
        ▼
   HDF5 database.h5 (Schema v2)              LanceDB vectors.lance/
   ├── global_descriptors/                    └── global_vectors table
   │   └── frame_poses  (N × 3 × 3)              (frame_id, vector[1024])
   ├── local_features/
   │   ├── keypoints     (N × 2048 × 2)
   │   ├── descriptors   (N × 2048 × 128)  fp16
   │   ├── coords_2d     (N × 2048 × 2)
   │   └── kp_counts     (N,)
   └── metadata/
       ├── num_frames, frame_width, frame_height
       ├── descriptor_dim, hdf5_schema = "v2"
       └── frame_index_map  (keyframe → slot mapping)
```

### 3.2 GPS Calibration & Propagation

```
 User clicks 3+ GCPs per anchor frame
        │
        ▼
 ┌───────────────────────────────────────────┐
 │  CalibrationDialog                        │
 │   pixel (x, y) + GPS (lat, lon) pairs     │
 │   ──▶  estimateAffine2D → AnchorCalibration│
 └─────────────────┬─────────────────────────┘
                   │
                   ▼
 ┌───────────────────────────────────────────┐
 │  CalibrationPropagationWorker             │
 │                                            │
 │  Phase 1: Prefetch all local features     │
 │  Phase 2: Build temporal edges            │
 │           (frame i ──H──▶ frame i+1)      │
 │  Phase 3: Detect loop closures            │
 │           (DINOv2 FAISS/LanceDB retrieval │
 │            + LightGlue matching)          │
 │  Phase 4: Fix anchor nodes,              │
 │           BFS initialize free nodes,      │
 │           Levenberg-Marquardt optimize     │
 │           (PoseGraphOptimizer, 5-DoF)     │
 │  Phase 5: Interpolate gaps (5-DoF PCHIP), │
 │           Save to HDF5 /calibration group │
 └─────────────────┬─────────────────────────┘
                   │
                   ▼
 HDF5 /calibration
 ├── frame_affine       (N × 2 × 3)  float64
 ├── frame_valid        (N,)          bool
 ├── frame_rmse         (N,)          float64
 ├── frame_disagreement (N,)          float64
 ├── frame_matches      (N,)          int32
 └── attrs: projection_json, anchors_json, version="3.0"
```

### 3.3 Real-Time Localization

```
 Drone Video Feed (or test file)
        │
        ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  RealtimeTrackingWorker.run()                                    │
 │                                                                  │
 │     for every frame:                                             │
 │       ├── emit frame_ready (→ VideoWidget)                       │
 │       │                                                          │
 │       ├── if keyframe (every N-th):                              │
 │       │     1. YOLO mask → static_mask                           │
 │       │     2. Localizer.localize_frame(frame_rgb, mask, dt)     │
 │       │        ├─ DINOv2 global desc (4 rotations if auto)      │
 │       │        ├─ FastRetrieval/LanceDB → top-K candidates      │
 │       │        ├─ ALIKED local features                          │
 │       │        ├─ LightGlue match → Homography (MAGSAC++)       │
 │       │        ├─ Center pixel → ref pixels → metric → GPS      │
 │       │        ├─ OutlierDetector.is_outlier()                   │
 │       │        ├─ TrajectoryFilter.update() (Kalman)             │
 │       │        └─ FOV polygon corners → GPS                     │
 │       │     3. Emit location_found(lat, lon, conf, inliers)    │
 │       │     4. Emit fov_found(polygon)                          │
 │       │     5. Save prev_gray + goodFeaturesToTrack()            │
 │       │                                                          │
 │       └── else (inter-frame):                                    │
 │             calcOpticalFlowPyrLK()                               │
 │             median (dx, dy) → localize_optical_flow()            │
 │             reuse last Homography + Affine matrices              │
 └──────────────────────────────────────────────────────────────────┘
```

---

## 4. Key Interfaces & Contracts

| Interface | Producer | Consumer | Data Shape |
|---|---|---|---|
| `FeatureExtractor.extract_features()` | `DatabaseBuilder`, `Localizer` | `FeatureMatcher` | `{keypoints: (K,2), descriptors: (K,D), global_desc: (1024,)}` |
| `FeatureMatcher.match()` | Matching stage | `Localizer`, `PropagationWorker` | `(mkpts_q: (M,2), mkpts_r: (M,2))` |
| `FastRetrieval.find_similar_frames()` | DINOv2 global search | `Localizer` | `[(frame_id, score), ...]` |
| `DatabaseLoader.get_local_features()` | HDF5 read (with LRU cache, max 200) | `Localizer`, `PropagationWorker` | `{keypoints, descriptors, coords_2d}` |
| `DatabaseLoader.get_frame_affine()` | `/calibration` group | `Localizer` | `np.ndarray (2,3) float64` or `None` |
| `GeometryTransforms.estimate_homography()` | OpenCV MAGSAC++ / PoseLib | `Localizer`, `PropagationWorker` | `(H: (3,3), mask: (N,1))` |
| `CoordinateConverter.metric_to_gps()` | pyproj Transformer | `Localizer` | `(lat: float, lon: float)` |
| `PoseGraphOptimizer.optimize()` | scipy.optimize.least_squares (TRF) | `PropagationWorker` | `{frame_id: affine_2x3}` |
| `ModelManager.load_<model>()` | torch.hub / ultralytics / LightGlue | All consumers | model object (thread-safe, locked) |

---

## 5. Configuration Hierarchy

`config/config.py` defines a single `AppConfig` Pydantic model:

```
AppConfig
├── dinov2           — descriptor_dim (1024), input_size (336)
├── database         — frame_step, prefetch_queue_size, use_lancedb, keyframe thresholds
├── localization     — min_matches, ransac_threshold, retrieval_top_k, auto_rotation, confidence weights
├── tracking         — Kalman noise params, outlier detector thresholds, process_fps
├── preprocessing    — CLAHE config, masking_strategy ("yolo" | "none")
├── gui              — video_fps, verify_display_mode
├── models           — per-model VRAM requirements, backend selection (git/torchscript/tensorrt)
│   ├── yolo, xfeat, aliked, superpoint, lightglue, dinov2, cesp
│   ├── vram_management — max_vram_ratio, default eviction threshold
│   └── performance    — torch_compile, fp16, log_level, debug_mode
├── projection       — WEB_MERCATOR default, anchor quality thresholds
├── homography       — backend ("opencv" | "poselib"), RANSAC params
└── graph_optimization — loop closure params, edge weights, LM iterations
```

Access pattern: `get_cfg(config, "dot.path", default)` — works with both dicts and Pydantic models.

---

## 6. Threading Model

| Thread | Type | Responsibilities |
|---|---|---|
| **Main (GUI)** | Qt Event Loop | UI rendering, signal dispatch, user interaction |
| **StartupWorker** | `QThread` | Background model prewarm at launch (`main.py:28`) |
| **DatabaseWorker** | `QThread` | Long-running DB build from video |
| **RealtimeTrackingWorker** | `QThread` | Frame decode → localize → emit results |
| **CalibrationPropagationWorker** | `QThread` | Graph build + LM optimization |
| **PanoramaWorker** | `QThread` | Video stitching |
| **PanoramaOverlayWorker** | `QThread` | Georeferenced overlay generation |
| **Prefetch (daemon)** | `threading.Thread` | Video frame decode ahead-of-time in `DatabaseBuilder` |
| **Pre-warm (daemon)** | `threading.Thread` | Load fallback models during tracking start |

**Synchronization:** `ModelManager._model_lock` (threading.Lock) protects all model load/unload operations against race conditions between prewarm and main-thread consumers.

---

## 7. Storage Formats

### HDF5 v2 Schema (`database.h5`)
- **Chunked arrays** with LZF compression; SWMR mode enabled
- Pre-allocated slots indexed by absolute `frame_id` (not sequential keyframe index)
- `kp_counts` array avoids zero-padding ambiguity

### LanceDB (`vectors.lance/`)
- IVF-PQ index built when `≥256` frames; cosine metric
- Replaces in-HDF5 global descriptors for faster ANN retrieval

### JSON files
- `project.json` — per-project metadata
- `calibration.json` — anchor list + projection metadata (version 2.2)
- `~/.drone_localizer/projects.json` — cross-session project registry
