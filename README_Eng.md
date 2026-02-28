# Drone Topometric Localization System

A professional system for topometric localization and visual navigation of drones in environments without a stable GPS signal. By leveraging modern Foundation Models and semantic analysis, the system provides high-precision coordinate determination that is robust to harsh changes in lighting, shadows, weather conditions, and seasons.

## 🎯 Key Features

- **Desktop GUI (PyQt6)**: A multi-threaded graphical interface for Windows, optimized for real-time operation.
- **Semantic Global Localization (DINOv2)**: Uses Meta's transformer model to understand the physical nature and geometry of objects. Allows the system to recognize terrain regardless of whether it is sunlit or in shadow.
- **Dynamic Object Filtering (YOLOv8-Seg)**: Automatic neural network masking of cars, people, and other moving objects during reference database creation, ensuring anchoring only to stable geometry (roads, buildings).
- **Adaptive Preprocessing (CLAHE)**: A local contrast equalization algorithm that "pulls out" textures and edges even from the deepest shadows for reliable local recognition.
- **Precise Feature Matching**: A combination of SuperPoint (keypoint extraction) and LightGlue (fast and robust match search).
- **Multi-Anchor Calibration & Wave Propagation**: Interactive assignment of GPS coordinates to several key frames, followed by automatic mathematical propagation (affine transformations) across all thousands of intermediate frames in the database.
- **Robust Trajectory Filtering**: A built-in Kalman Filter for trajectory smoothing and a statistical anomaly detector (Z-score + Speed limit) to eliminate false localization jumps.
- **Interactive Map & Panoramas**: Display of the drone, its trajectory, and field of view (FOV) on a Leaflet map. Generation of panoramas from video and their automatic geospatial overlay on the satellite map.

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

# Install hloc directly from GitHub (REQUIRED)
pip install git+https://github.com/cvg/Hierarchical-Localization.git
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
├── gui/              # PyQt6 graphical interface
├── workers/          # QThread background threads (tracking, DB, panoramas)
├── models/           # Neural network wrappers (YOLO, SuperPoint, NetVLAD, LightGlue)
├── database/         # HDF5 generation and reading
├── localization/     # Recognition pipeline
├── geometry/         # Geometric transformations (affine, homography)
├── calibration/      # Multi-anchor calibration and wave propagation
├── tracking/         # Kalman filter and outlier detection (Z-test)
└── utils/            # Helper functions and logging
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