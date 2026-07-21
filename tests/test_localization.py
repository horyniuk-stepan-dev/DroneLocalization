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
        # Confidence reads DB-QA arrays directly (result_builder.compute_confidence).
        self.mock_db.frame_rmse = None
        self.mock_db.frame_disagreement = None

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
        """Current API: _compute_confidence(candidate_id, inliers, total_matches, rmse_val).

        Confidence = 0.3*stability + 0.4*inlier_score + 0.3*match_score, where
        stability comes from DB-QA (frame_rmse / frame_disagreement) and
        inlier_score = min(1, inliers / confidence_max_inliers[=80]).
        """
        # High: many inliers, low RMSE, clean DB-QA -> conf well above 0.8.
        self.mock_db.frame_rmse = np.full(100, 0.5, dtype=np.float32)
        self.mock_db.frame_disagreement = np.full(100, 0.2, dtype=np.float32)
        confidence = self.localizer._compute_confidence(
            best_candidate_id=0, best_inliers=150, total_matches=160, rmse_val=0.5
        )
        self.assertGreater(confidence, 0.8)  # re-derived: ~0.956

        # Low: few inliers, high RMSE, poor DB-QA -> conf well below 0.4.
        self.mock_db.frame_rmse = np.full(100, 10.0, dtype=np.float32)
        self.mock_db.frame_disagreement = np.full(100, 5.0, dtype=np.float32)
        confidence_low = self.localizer._compute_confidence(
            best_candidate_id=0, best_inliers=12, total_matches=120, rmse_val=20.0
        )
        self.assertLess(confidence_low, 0.4)  # re-derived: ~0.095

    def test_localize_optical_flow(self):
        # Fake a successful keyframe state (OF is applied relative to it).
        self.localizer._last_state = {
            "H": np.eye(3, dtype=np.float32),
            "affine": np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 50.0]], dtype=np.float32),
            "candidate_id": 5,
            "global_angle": 0,
            "scale": 1.0,
        }
        # Current API reads calibration.converter.metric_to_gps.
        self.localizer.calibration.converter.metric_to_gps.return_value = (48.0, 35.0)

        # Current signature requires the original frame size (rot_width/rot_height).
        res = self.localizer.localize_optical_flow(
            dx_px=10.0, dy_px=-5.0, dt=0.03, rot_width=1920, rot_height=1080
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["is_of"], True)
        self.assertIn("lat", res)
        self.assertIn("lon", res)


if __name__ == "__main__":
    unittest.main()
