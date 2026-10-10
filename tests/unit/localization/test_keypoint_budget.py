"""src/localization/keypoint_budget.py: strongest-first keypoint slice."""

import numpy as np

from src.localization.keypoint_budget import top_keypoints


def _features(n: int) -> dict:
    kp = np.arange(2 * n, dtype=np.float32).reshape(n, 2)
    return {
        "keypoints": kp,
        "descriptors": np.arange(n * 4, dtype=np.float32).reshape(n, 4),
        "coords_2d": kp.copy(),
        "image_size": np.array([2, 3], dtype=np.int32),  # per image, len 2
    }


def test_no_budget_or_within_budget_returns_the_same_dict():
    f = _features(10)
    assert top_keypoints(f, 0) is f
    assert top_keypoints(f, -1) is f
    assert top_keypoints(f, 10) is f
    assert top_keypoints(f, 50) is f
    assert top_keypoints(None, 5) is None
    assert top_keypoints({}, 5) == {}


def test_slice_keeps_the_first_entries_and_leaves_the_input_alone():
    f = _features(10)
    out = top_keypoints(f, 4)
    assert out is not f
    np.testing.assert_array_equal(out["keypoints"], f["keypoints"][:4])
    np.testing.assert_array_equal(out["descriptors"], f["descriptors"][:4])
    np.testing.assert_array_equal(out["coords_2d"], f["coords_2d"][:4])
    np.testing.assert_array_equal(out["image_size"], [2, 3])
    assert len(f["keypoints"]) == 10 and len(f["descriptors"]) == 10  # cache entry intact


def test_per_image_value_with_matching_length_is_not_sliced():
    """image_size has length 2: with 2 keypoints and limit 1 it must survive."""
    f = _features(2)
    out = top_keypoints(f, 1)
    assert len(out["keypoints"]) == 1
    np.testing.assert_array_equal(out["image_size"], [2, 3])
