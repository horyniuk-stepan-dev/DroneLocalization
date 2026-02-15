# User Guide

## Getting Started

### Creating a New Mission

1. Launch the application
2. Click "New Mission"
3. Enter mission name
4. Select reference video file
5. Set flight altitude (meters)
6. Click "Create"

### Building Database

1. Wait for video processing
2. Monitor progress bar
3. Database will be saved automatically
4. Status will show "Complete" when done

### GPS Calibration

1. Load database
2. Click "Calibration" → "GPS Calibration"
3. Click on 3+ landmarks in video
4. Click on same landmarks in satellite map
5. Click "Calculate"
6. Verify RMSE is acceptable (<10m)
7. Click "Save"

### Real-time Tracking

1. Connect drone video feed
2. Load calibrated database
3. Click "Start Tracking"
4. Watch real-time localization on map
5. Click "Stop Tracking" when done

## Interface Overview

### Main Window
- **Center**: Video display with overlays
- **Left**: Control panel
- **Right**: Interactive map
- **Bottom**: Status bar

### Control Panel
- **Mission**: Create, load databases
- **Tracking**: Start/stop tracking
- **Calibration**: GPS calibration tools
- **Status**: Progress and messages

### Map View
- **Markers**: Current position
- **Trajectory**: Path history
- **Layers**: Satellite, street views

## Tips

- Use high-quality reference videos
- Fly at consistent altitude
- Choose distinctive landmarks for calibration
- Monitor confidence scores during tracking
- Save trajectories for later analysis
