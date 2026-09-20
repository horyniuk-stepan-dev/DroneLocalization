from unittest.mock import MagicMock

import numpy as np
import torch

from src.models.wrappers.feature_extractor import FeatureExtractor


def _extractor_with_features(keypoints: list[list[float]]) -> FeatureExtractor:
    extractor = FeatureExtractor.__new__(FeatureExtractor)
    extractor.config = {}
    extractor.preprocessor = MagicMock()
    extractor.preprocessor.preprocess.side_effect = lambda image: image
    extractor._upload_chw = MagicMock(return_value=torch.zeros((1, 3, 20, 20)))
    extractor.local_model = MagicMock()
    extractor.local_model.return_value = {
        "keypoints": torch.tensor([keypoints], dtype=torch.float32),
        "descriptors": torch.arange(len(keypoints) * 4, dtype=torch.float32).reshape(
            1, len(keypoints), 4
        ),
    }
    return extractor


def test_fully_dynamic_mask_returns_no_local_features():
    extractor = _extractor_with_features([[2.0, 3.0], [10.0, 11.0]])
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    dynamic_mask = np.zeros((20, 20), dtype=np.uint8)

    features = extractor.extract_local_features(image, dynamic_mask)

    assert features["keypoints"].shape == (0, 2)
    assert features["descriptors"].shape == (0, 4)
    assert features["coords_2d"].shape == (0, 2)


def test_static_mask_keeps_only_keypoints_on_static_pixels():
    extractor = _extractor_with_features([[2.0, 3.0], [10.0, 11.0]])
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    static_mask = np.zeros((20, 20), dtype=np.uint8)
    static_mask[3, 2] = 255

    features = extractor.extract_local_features(image, static_mask)

    np.testing.assert_array_equal(features["keypoints"], [[2.0, 3.0]])
    assert features["descriptors"].shape == (1, 4)
