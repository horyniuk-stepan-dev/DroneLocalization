import pytest
import numpy as np
from src.tracking.object_tracker import ObjectTracker, TrackedObject

@pytest.fixture
def tracker():
    config = {
        "track_activation_threshold": 0.25,
        "lost_track_buffer": 30,
        "minimum_matching_threshold": 0.8,
    }
    return ObjectTracker(config)

def test_tracker_initialization(tracker):
    assert tracker is not None
    assert tracker.tracker is not None

def test_tracker_update_empty(tracker):
    detections = []
    tracked = tracker.update(detections, (1080, 1920))
    assert len(tracked) == 0

def test_tracker_update_with_detections(tracker):
    detections = [
        {"class_id": 0, "confidence": 0.9, "bbox": [100.0, 100.0, 200.0, 300.0]},
        {"class_id": 2, "confidence": 0.85, "bbox": [500.0, 500.0, 600.0, 600.0]}
    ]
    
    tracked = tracker.update(detections, (1080, 1920))
    
    # Depending on supervision version and exact logic, it might return them immediately
    # or wait for a few frames. With track_activation_threshold=0.25 and confidence 0.9, it should track.
    assert len(tracked) == 2
    
    # Check classes
    classes = [t.class_id for t in tracked]
    assert 0 in classes
    assert 2 in classes
    
    # Check centers
    for t in tracked:
        if t.class_id == 0:
            assert t.center_px == (150.0, 200.0)
            assert t.class_name == "person"
        elif t.class_id == 2:
            assert t.center_px == (550.0, 550.0)
            assert t.class_name == "car"
            
def test_tracker_reset(tracker):
    # Process some detections
    detections = [{"class_id": 0, "confidence": 0.9, "bbox": [100, 100, 200, 200]}]
    tracker.update(detections, (1080, 1920))
    
    old_tracker = tracker.tracker
    tracker.reset()
    new_tracker = tracker.tracker
    
    assert old_tracker is not new_tracker
