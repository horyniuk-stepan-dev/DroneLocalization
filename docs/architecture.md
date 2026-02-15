# Technical Architecture

See detailed architecture documents:
- corrected_architecture.md - Full technical specification
- architecture_description.md - High-level overview

## Key Components

### GUI Layer (PyQt6)
- QMainWindow with dock widgets
- QGraphicsView for video display
- QWebEngineView for Leaflet.js map
- QThread workers for background tasks

### Processing Layer
- YOLOv8-Seg for object segmentation
- SuperPoint for keypoint extraction
- NetVLAD for global descriptors
- LightGlue for feature matching
- Depth-Anything for depth estimation

### Storage Layer
- HDF5 for topometric database
- Hierarchical structure
- Lazy loading for memory efficiency

### Coordination Layer
- QThread for parallel processing
- pyqtSignal for thread-safe communication
- GPU memory management
- Kalman filtering for trajectory smoothing
