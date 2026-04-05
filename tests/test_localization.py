import unittest
from unittest.mock import MagicMock

import numpy as np

from src.localization.localizer import Localizer


class TestLocalizationConfidence(unittest.TestCase):
    def setUp(self):
        # Mock database loader
        self.mock_db = MagicMock()
        self.mock_db.metadata = {"frame_width": 1920, "frame_height": 1080}
        self.mock_db.get_num_frames.return_value = 100
        self.mock_db.is_propagated = True
        self.mock_db.frame_affine = [np.eye(2, 3) for _ in range(100)]
        self.mock_db.frame_valid = [True for _ in range(100)]
        self.mock_db.global_descriptors = np.random.randn(100, 1024).astype(np.float32)

        # Mock feature extractor
        self.mock_extractor = MagicMock()

        # Initialize Localizer
        self.localizer = Localizer(
            database=self.mock_db,
            feature_extractor=self.mock_extractor,
            matcher=MagicMock(),
            calibration=MagicMock(),  # Mock calibration
        )

    def test_compute_confidence(self):
        # High confidence scenario
        inliers = 150
        max_inliers = 300
        rmse = 0.5
        features_count = 500

        confidence = self.localizer._compute_confidence(inliers, max_inliers, rmse, features_count)
        self.assertGreater(confidence, 0.8)

        # Low confidence scenario (low inliers, high rmse)
        inliers = 12
        max_inliers = 150
        rmse = 5.0
        features_count = 500

        confidence_low = self.localizer._compute_confidence(
            inliers, max_inliers, rmse, features_count
        )
        self.assertLess(confidence_low, 0.4)

    def test_localize_optical_flow(self):
        # Initialize the state to fake successful keyframe match
        self.localizer._last_state = {
            "H": np.eye(3, dtype=np.float32),
            "affine": np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 50.0]], dtype=np.float32),
            "candidate_id": 5,
        }
        self.localizer.converter = MagicMock()
        self.localizer.converter.local_to_wgs84.return_value = (48.0, 35.0)

        # Apply optical flow pixel shift
        dx_px = 10.0
        dy_px = -5.0
        dt = 0.03

        res = self.localizer.localize_optical_flow(dx_px, dy_px, dt)
        self.assertTrue(res["success"])
        self.assertEqual(res["is_of"], True)
        self.assertIn("lat", res)
        self.assertIn("lon", res)


if __name__ == "__main__":
    unittest.main()
