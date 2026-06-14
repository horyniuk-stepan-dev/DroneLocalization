# Drone Topometric Localization System

A professional system for topometric localization and visual navigation of drones in environments without a stable GPS signal. By leveraging modern Foundation Models and semantic analysis, the system provides high-precision coordinate determination that is robust to harsh changes in lighting, shadows, weather conditions, and seasons.

## 🎯 Key Features

- **Desktop GUI (PyQt6)**: Multi-threaded graphical interface for Windows, optimized for real-time operation.
- **Headless Mode**: Server-friendly deployment with WebSocket + REST API for external integrations.
- **Semantic Global Localization (DINOv2)**: Leverages Meta's Foundation Model for global descriptors resistant to lighting, shadows, and seasonal changes.
- **Dynamic Object Filtering (YOLOv11-Seg)**: Automatic neural masking of moving objects (cars, people) to ensure anchoring only to stable geometry.
- **Adaptive Preprocessing (CLAHE)**: Local contrast equalization to extract textures from deep shadows.
- **Hybrid Feature Matching**: Uses **ALIKED + LightGlue** for high-precision keypoint matching in challenging scenarios.
- **Depth Estimation (Depth-Anything-V2)**: Monocular depth estimation to support scale-aware localization.
- **Multi-Anchor Calibration**: Interactive GPS anchoring with automatic \"wave\" propagation of coordinates via pose graph optimization (5-DoF Levenberg-Marquardt).
- **Intelligent Tracking**: Trajectory smoothing via **Kalman Filter** and anomaly detection using Z-score.
- **Interactive Map**: Real-time visualization of Drone FOV (Field of View) and path on a Leaflet-based map.
- **Result Export**: CSV, GeoJSON, and KML export of localization results.

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

```powershell
# Download all required model weights automatically
python scripts/download_models.py
```

Alternatively, download `models.zip` manually from [Google Drive](https://drive.google.com/drive/folders/1qyO9AtUNkmkHXswvCbNkYcoTv8ChBbyP?usp=sharing) and extract to `models/weights/`.

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
| `--ws-port` | 8765 | WebSocket server port |
| `--rest-port` | 8080 | REST API server port |

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
3. Find the starting frame of the route, click on 3+ landmarks in the video and enter their real GPS coordinates. Add the anchor.
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
├── core/             # Project lifecycle, headless runner, result export
├── models/           # AI model wrappers (DINOv2, ALIKED, YOLOv11, LightGlue, TensorRT)
├── database/         # HDF5 v2 + LanceDB management (Builder, Loader)
├── localization/     # Core pipeline (Localizer, FeatureMatcher, retrieval)
├── geometry/         # Math (Affine, Homography, PoseGraph, UTM coordinates)
├── calibration/      # GPS anchor management and coordinate propagation
├── tracking/         # Kalman Filter and Outlier Detector
├── depth/            # Monocular depth estimation (Depth-Anything-V2)
├── network/          # WebSocket server, REST API, coordinates broker
├── video/            # Video frame decoding utilities
├── workers/          # QThread background tasks (tracking, DB, calibration, panoramas)
├── gui/              # PyQt6 desktop interface (MainWindow, mixins, widgets, dialogs)
└── utils/            # Logging (Loguru), config, CLAHE preprocessing, telemetry
```

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

### Compiling to .exe

```powershell
python scripts/build_executable.py
```

## 📧 Contact

For questions and support, please open an Issue on GitHub.