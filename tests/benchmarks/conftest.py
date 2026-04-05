import os

import numpy as np
import pytest

# Create the fixtures directory if it doesn't exist
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)


@pytest.fixture(scope="session")
def dummy_video_path():
    """
    Returns path to a dummy video. Generates one if it doesn't exist.
    """
    path = os.path.join(FIXTURES_DIR, "dummy_video.mp4")
    if not os.path.exists(path):
        import cv2

        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
        # Generate 50 frames
        for _ in range(50):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
    return path


@pytest.fixture(scope="session")
def long_dummy_video_path():
    """
    Returns path to a longer dummy video (300 frames). Generates one if it doesn't exist.
    """
    path = os.path.join(FIXTURES_DIR, "long_dummy_video.mp4")
    if not os.path.exists(path):
        import cv2

        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
        # Generate 300 frames
        for _ in range(300):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
    return path
