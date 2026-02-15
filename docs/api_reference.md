# API Reference

## Core Classes

### MainWindow
Main application window.

### DatabaseBuilder
Builds topometric database from video.

### Localizer
Localizes frames using database.

### GPSCalibration
Manages GPS coordinate calibration.

## Worker Threads

### DatabaseGenerationWorker
Background thread for database creation.

### RealtimeTrackingWorker
Background thread for real-time localization.

## Models

### ModelManager
Loads and manages neural networks.

### YOLOWrapper
YOLOv8 segmentation wrapper.

### FeatureExtractor
SuperPoint + NetVLAD feature extraction.

## Utilities

### opencv_to_qpixmap
Convert OpenCV image to Qt format.

### CoordinateConverter
GPS/metric coordinate conversions.

### TrajectoryFilter
Kalman filter for trajectory smoothing.
