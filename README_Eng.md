# Drone Topometric Localization System

A professional system for topometric localization and visual navigation of drones in environments without a stable GPS signal. By leveraging modern Foundation Models and semantic analysis, the system provides high-precision coordinate determination that is robust to harsh changes in lighting, shadows, weather conditions, and seasons.

## 🎯 Key Features

- **Desktop GUI (PyQt6)**: Multi-threaded graphical interface for Windows, optimized for real-time operation.
- **Headless Mode**: Server-friendly deployment with WebSocket + REST API for external integrations.
- **Semantic Global Localization (DINOv3 / DINOv2)**: Meta's Foundation Model for global descriptors resistant to lighting, shadow, and seasonal changes. Default backend is **DINOv3 ViT-L/16** pretrained on 493M satellite images (`models.global_descriptor.backend = "dinov3"`); DINOv2 ViT-L/14 remains available as an alternative backend.
- **VLAD Aggregation (AnyLoc, optional)**: Training-free VLAD aggregation of DINOv3 patch tokens instead of the CLS descriptor — enabled in config, requires a vocabulary (`scripts/build_vlad_vocab.py`) and a database rebuild.
- **Dynamic Object Filtering (YOLOv11-Seg)**: Automatic neural masking of moving objects (cars, people) to ensure anchoring only to stable geometry.
- **Adaptive Preprocessing (CLAHE)**: Local contrast equalization to extract textures from deep shadows.
- **Hybrid Feature Matching**: Uses **ALIKED + LightGlue** for high-precision keypoint matching in challenging scenarios; **RDD** is available as an alternative local feature detector.
- **Depth Estimation (Depth-Anything-V2)**: Monocular depth estimation to support scale-aware localization.
- **Multi-Anchor Calibration**: Interactive GPS anchoring with automatic \"wave\" propagation of coordinates via pose graph optimization (5-DoF Levenberg-Marquardt).
- **Intelligent Tracking**: Trajectory smoothing via **Kalman Filter**, a fixed-lag smoother (servo mode), and anomaly detection using Z-score.
- **Object Tracking**: Detection and projection of moving objects from the frame into GPS coordinates.
- **Multi-Database / Multi-Calibration**: Several databases and calibrations loaded at once, with the active one switched during localization.
- **Interactive Map**: Real-time visualization of Drone FOV (Field of View) and path on a Leaflet-based map.
- **Result Export**: CSV, GeoJSON, and KML export of localization results.
- **Encryption at Rest**: Optional passphrase encryption of project files (databases, calibration, video), decrypted in memory on load.
- **Hardware Auto-Tuning**: Automatic GPU/CPU profiling that tunes batch sizes, thread counts, and the VRAM budget (`models.performance.auto_tune`).

## 📋 System Requirements

### Hardware Requirements
- **GPU**: NVIDIA with CUDA support (minimum **4GB VRAM**, e.g., GTX 1650 or better)
- **CPU**: 6+ cores
- **RAM**: 16 GB (32 GB recommended for large databases)
- **Storage**: SSD (recommended)

### Software Requirements
- Python 3.10–3.11
- PyTorch ≥ 2.2.0 (CUDA 12.x)
- Windows 10/11 (primary platform)

## 🚀 Quick Start

### 1. Installation

```powershell
# Clone the repository
git clone https://github.com/horyniuk-stepan-dev/DroneLocalization.git
cd DroneLocalization

# Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install the project and dependencies (editable mode)
pip install -e .

# (Optional) Install dev dependencies (pytest, ruff, ...)
pip install -e ".[dev]"

# (Optional) Install TensorRT acceleration for YOLO
pip install -e ".[tensorrt]"
```

### 1.1 Installing RDD (Robust Deformable Detector)
The RDD source code is already included in the repository under `third_party/rdd/`. It is recommended to compile the custom CUDA operators for performance:

```powershell
# Install RDD dependencies
pip install -r third_party/rdd/requirements.txt

# Compile custom CUDA operators (recommended for speed)
cd third_party/rdd/RDD/models/ops
pip install -e . --no-build-isolation
```

### 1.2 Installing Depth-Anything-V2
The depth estimation module requires cloning Depth-Anything-V2 into `third_party/`:

```powershell
# Clone Depth-Anything-V2 into third_party
git clone https://github.com/DepthAnything/Depth-Anything-V2 third_party/Depth-Anything-V2

# Install its dependencies
pip install -r third_party/Depth-Anything-V2/requirements.txt
```

### 1.3 Downloading Model Weights

**Primary route — the `models.zip` archive on Google Drive.** Automatic download does
not cover everything: `scripts/download_models.py` only fetches `yolo11n-seg.pt` and
`depth_anything_v2_vits.pth`. RDD weights and **DINOv3** cannot be fetched that way —
`facebook/dinov3-vitl16-pretrain-sat493m` is a gated HuggingFace repository and requires
individually approved access.

```powershell
# Extract into the repository root — a models/ folder should appear
Expand-Archive models.zip -DestinationPath .
```

Archive [here](https://drive.google.com/drive/folders/1qyO9AtUNkmkHXswvCbNkYcoTv8ChBbyP?usp=sharing).
After extraction the layout should be:

```
models/
├── yolo11n-seg.pt / .onnx / .engine   # dynamic object masking
├── depth_anything_v2_vits.pth         # depth estimation
├── RDD-v*.pth, RDD_lg-v*.pth          # alternative feature detector
├── vlad_vocab_c32_p256.npz            # VLAD vocabulary (optional)
└── .cache/                            # torch.hub + HuggingFace cache (DINOv2/DINOv3)
```

> `models/` is the **single** storage root for weights: `config.paths.ensure_model_cache_env`
> redirects `TORCH_HOME`, `HF_HOME`, and `HUGGINGFACE_HUB_CACHE` into `models/.cache/`,
> so nothing lands in `C:\Users\<you>\.cache`.

**If you fetch DINOv3 yourself** (instead of using the cache from the archive):

1. Request access to `facebook/dinov3-vitl16-pretrain-sat493m` on HuggingFace and wait for approval.
2. Log in so the token is stored in the project cache, otherwise you will get a 401:

```powershell
$env:HF_HOME = "$PWD\models\.cache\huggingface"
hf auth login          # older huggingface_hub versions: huggingface-cli login
```

> **Security:** DINOv3 is loaded with `trust_remote_code=True`. Pin
> `models.global_descriptor.dinov3.hf_revision` to a commit hash so an upstream code change
> cannot turn into arbitrary code execution on your machine. Empty means latest.

Partial download (YOLO + Depth-Anything) is still available:

```powershell
python scripts/download_models.py
```

> **Note:** PyTorch with CUDA support must be installed separately following the
> [official instructions](https://pytorch.org/get-started/locally/), as the required
> version depends on your GPU and driver.

### 2. Running the Application

```powershell
# Launch GUI
python main.py

# Run in headless mode (WebSocket + REST API)
python main.py --headless --project /path/to/project --source /path/to/video.mp4
python main.py --headless --project /path/to/project --source rtsp://drone-ip/stream
```

**Headless options:**

| Flag | Default | Description |
|---|---|---|
| `--project` | required | Path to the project directory |
| `--source` | required | Video file path or RTSP/HTTP stream URL |
| `--ws-port` | from config (`network_api.ws_port`) | WebSocket server port |
| `--rest-port` | from config (`network_api.rest_port`) | REST API server port |
| `--supervise` | off | Run the pipeline in a child process and auto-restart it on crash |
| `--max-restarts` | 0 (unlimited) | Stop after N restarts (with `--supervise` only) |

## 📖 Workflow

### Stage 1: Database Creation

1. Launch the application.
2. Select "Create New Mission".
3. Load the reference video.
4. Set the flight altitude (for scaling).
5. Start processing.
6. Wait for completion (progress indicator).

### Stage 2: GPS Calibration

1. Open the created database.
2. Go to menu "Calibration" → "Add Anchor...".
3. The dialog automatically opens the `*_keypoints.mp4` video (its frame numbering matches the database; the original flight video is converted automatically). Find the starting frame of the route, click on 4+ landmarks (5–6 recommended, spread across the whole frame) and enter their real GPS coordinates. Add the anchor.
4. Repeat the procedure for a frame in the middle of the route and for the final frame.
5. Click "Done — Run Propagation". The application will automatically calculate coordinates for all thousands of intermediate frames using the LightGlue neural network and pose graph optimization.

### Stage 3: Real-Time Localization

1. Connect the drone or load a test flight video.
2. Load the calibrated database (with propagation already completed).
3. Click "Start Tracking".
4. Observe precise localization on the map in real time with trajectory smoothing.

### Stage 4: Panoramas (Optional)

1. Click "Generate Panorama from Video" to create a wide image of the area.
2. Click "Overlay Panorama on Map". The system will tile the panorama, find its coordinates via neural networks, and display it on top of the satellite map.

## 🏗️ Architecture

```
src/
├── core/             # Project lifecycle, headless runner, project registry, result export
├── models/           # AI model wrappers (DINOv3/DINOv2, ALIKED, RDD, YOLOv11, LightGlue, VLAD, TensorRT)
├── database/         # HDF5 v2 + LanceDB (Builder, Loader, keyframe selector, spatial index)
├── localization/     # Core pipeline (Localizer, matcher, retrieval, patchify, verification)
├── geometry/         # Math (Affine, Homography, 5-DoF pose_graph, UTM coordinates, GSD)
├── calibration/      # GPS anchor management and coordinate propagation
├── tracking/         # Kalman Filter, fixed-lag smoother, Outlier Detector, object tracker
├── depth/            # Monocular depth estimation (Depth-Anything-V2)
├── network/          # WebSocket server, REST API, coordinates broker
├── security/         # Encryption at rest, project scanning
├── video/            # Video frame decoding utilities
├── workers/          # QThread background tasks (tracking, DB, propagation, panoramas, encryption)
├── gui/              # PyQt6 desktop interface (MainWindow, mixins, widgets, dialogs)
└── utils/            # Logging (Loguru), CLAHE preprocessing, hardware profile, telemetry, fault injection

config/               # Domain-split Pydantic config: models, database, localization, graph, app, access
```

> Detailed flow diagrams (build / calibration / localization) live in `docs/architecture.md`;
> the pose-graph math is in `docs/POSE_GRAPH_MATH.md`.

## 🔧 Development

### Running Tests

```powershell
# Requires: pip install -e ".[dev]"
pytest tests/ -v
```

### Linting

```powershell
ruff check src/
ruff format src/
```

### Compiling TensorRT Engine

```powershell
python scripts/compile_dinov2_trt.py
```

### Useful Scripts

```powershell
python scripts/benchmark_hardware.py     # hardware profile and recommended settings
python scripts/build_vlad_vocab.py       # build a VLAD vocabulary from reference frames
python scripts/encrypt_project.py        # produce an encrypted copy of a project
python scripts/soak_test.py              # long-running stress test with injected faults
python scripts/validate_vs_ground_truth.py  # compare against simulator ground truth
```

### Compiling to .exe

```powershell
python scripts/build_executable.py
```

## 📧 Contact

For questions and support, please open an Issue on GitHub.