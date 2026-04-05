import numpy as np
import pytest

from src.localization.localizer import Localizer


# Dummy classes to mock dependencies for pure Localizer benchmarking
class DummyDatabase:
    def __init__(self):
        self.global_descriptors = np.random.randn(100, 1024).astype(np.float32)

    def get_local_features(self, candidate_id):
        return {
            "keypoints": np.random.rand(100, 2).astype(np.float32),
            "descriptors": np.random.rand(100, 128).astype(np.float32),
        }

    def get_frame_affine(self, candidate_id):
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)


class DummyExtractor:
    def extract_global_descriptor(self, frame):
        return np.random.rand(1, 384).astype(np.float32)

    def extract_local_features(self, frame, static_mask=None):
        return {
            "keypoints": np.random.rand(50, 2).astype(np.float32),
            "descriptors": np.random.rand(50, 128).astype(np.float32),
        }


class DummyMatcher:
    def match(self, q, r):
        # Return 15 dummy matches
        mkpts_q = np.random.rand(15, 2).astype(np.float32) * 100
        mkpts_r = np.random.rand(15, 2).astype(np.float32) * 100
        return mkpts_q, mkpts_r


class DummyConverter:
    def metric_to_gps(self, x, y):
        return 49.0, 24.0


class DummyCalibration:
    def __init__(self):
        self.converter = DummyConverter()


@pytest.fixture
def localizer_deps():
    return {
        "database": DummyDatabase(),
        "feature_extractor": DummyExtractor(),
        "matcher": DummyMatcher(),
        "calibration": DummyCalibration(),
    }


def test_benchmark_tracking(benchmark, localizer_deps):
    """
    Benchmarks the tracking/localization loop performance.
    """
    localizer = Localizer(
        database=localizer_deps["database"],
        feature_extractor=localizer_deps["feature_extractor"],
        matcher=localizer_deps["matcher"],
        calibration=localizer_deps["calibration"],
        config={},
    )

    # Mock retrieval to avoid loading Faiss/DINO
    localizer.retriever.find_similar_frames = lambda desc, top_k: [(1, 0.95)]

    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def run_tracking():
        for _ in range(10):  # simulate 10 frames tracking
            localizer.localize_frame(dummy_frame)

    # Measure the performance of 10 consecutive localize_frame calls
    benchmark(run_tracking)
