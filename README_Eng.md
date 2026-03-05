# Drone Topometric Localization System

A professional system for topometric localization and visual navigation of drones in environments without a stable GPS signal. By leveraging modern Foundation Models and semantic analysis, the system provides high-precision coordinate determination that is robust to harsh changes in lighting, shadows, weather conditions, and seasons.

## 🎯 Key Features

- **Desktop GUI (PyQt6)**: Multi-threaded graphical interface for Windows, optimized for real-time operation.
- **Semantic Global Localization (DINOv2)**: Leverages Meta's Foundation Model for global descriptors resistant to lighting, shadows, and seasonal changes.
- **Dynamic Object Filtering (YOLOv11-Seg)**: Automatic neural masking of moving objects (cars, people) to ensure anchoring only to stable geometry.
- **Adaptive Preprocessing (CLAHE)**: Local contrast equalization to extract textures from deep shadows.
- **Hybrid Feature Matching**: Uses **XFeat** for speed or **SuperPoint + LightGlue** for maximum precision in challenging scenarios.
- **Multi-Anchor Calibration**: Interactive GPS anchoring with automatic "wave" propagation of coordinates via affine transforms.
- **Intelligent Tracking**: Trajectory smoothing via **Kalman Filter** and anomaly detection using Z-score.
- **Interactive Map**: Real-time visualization of Drone FOV (Field of View) and path on a Leaflet-based map.

## 📋 System Requirements

### Hardware Requirements
- **GPU**: NVIDIA with CUDA support (minimum **4GB VRAM**, e.g., GTX 1650 or better)
- **CPU**: 6+ cores
- **RAM**: 16 GB (32 GB recommended for large databases)
- **Storage**: SSD (recommended; HDF5 databases take minimal space thanks to the 384-dimensional descriptor size)

### Software Requirements
- Python 3.11
- PyTorch 2.2.0 (CUDA 12.1)
- Windows 10/11 (primary platform)

## 🚀 Quick Start

### 1. Installation

```powershell
# Clone the repository
git clone <repository-url>
cd DroneLocalization

# Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

```

### 2. Running the Application

```powershell
# Launch GUI
python main.py
```

## 📖 Workflow

### Stage 1: Database Creation

1. Launch the application.
2. In the app, select "Create New Mission".
3. Load the reference video.
4. Set the flight altitude (for scaling).
5. Start processing.
6. Wait for completion (progress indicator).

### Stage 2: GPS Calibration

1. Open the created database.
2. Go to menu "Calibration" → "Add Anchor...".
3. Find the starting frame of the route, click on 3+ landmarks in the video and enter their real GPS coordinates. Add the anchor.
4. Repeat the procedure for a frame in the middle of the route and for the final frame.
5. Click "Done — Run Propagation". The application will automatically calculate coordinates for all thousands of intermediate frames using the LightGlue neural network.

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
├── gui/              # PyQt6 GUI and map components
├── workers/          # QThread background threads (tracking, DB, panoramas)
├── models/           # AI wrappers (YOLOv11, DINOv2, XFeat, LightGlue)
├── database/         # HDF5 management (Builder, Loader)
├── localization/     # Core pipeline (Localizer, Matcher)
├── geometry/         # Math (Affine, Homography, UTM coordinates)
├── calibration/      # GPS Calibration and propagation system
├── tracking/         # Kalman Filter and Outlier Detector
└── utils/            # Logging, config, and utility scripts
```

## 🔧 Development

### Running Tests

```powershell
pytest tests/ -v
```

### Compiling to .exe

```powershell
# PyInstaller
python scripts/build_executable.py
```

## 📧 Contact

For questions and support, please open an Issue on GitHub.