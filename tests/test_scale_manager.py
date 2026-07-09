"""Unit tests for ScaleManager — GSD-ratio estimation & frame normalization."""

import numpy as np
import pytest

from src.localization.scale_manager import CropInfo, ScaleManager


class TestScaleManagerCandidates:
    """Test candidate generation (prior vs pyramid)."""

    def test_no_prior_returns_full_pyramid(self):
        sm = ScaleManager()
        candidates = sm.candidates()
        assert len(candidates) == 5
        assert 1.0 in candidates

    def test_prior_returns_single_candidate(self):
        sm = ScaleManager()
        sm._prior = 1.3
        assert sm.candidates() == [1.3]

    def test_depth_hint_reorders_pyramid(self):
        sm = ScaleManager()
        sm._depth_hint = 2.0
        candidates = sm.candidates()
        # 2.0 should be first (closest to hint)
        assert candidates[0] == 2.0

    def test_invalidate_resets_prior(self):
        sm = ScaleManager()
        sm._prior = 1.5
        sm.invalidate()
        assert sm.prior is None
        assert len(sm.candidates()) == 5


class TestScaleManagerNormalize:
    """Test frame normalization (crop/resize)."""

    def _make_frame(self, w=640, h=480):
        return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    def test_r_near_one_is_noop(self):
        sm = ScaleManager()
        frame = self._make_frame()
        out, info = sm.normalize(frame, 1.0)
        assert out is frame  # same object
        assert info.resize_scale == 1.0

    def test_r_in_tolerance_band_is_noop(self):
        sm = ScaleManager()
        frame = self._make_frame()
        out, info = sm.normalize(frame, 0.9)
        assert out is frame
        assert info.resize_scale == 1.0

    def test_r_greater_than_one_crops_and_upscales(self):
        sm = ScaleManager()
        frame = self._make_frame(640, 480)
        out, info = sm.normalize(frame, 2.0)
        # Output should be same size (upscaled back)
        assert out.shape == (480, 640, 3)
        # Crop was applied
        assert info.crop_w < 640
        assert info.crop_h < 480
        assert info.resize_scale > 1.0

    def test_r_less_than_one_downscales(self):
        sm = ScaleManager()
        frame = self._make_frame(640, 480)
        out, info = sm.normalize(frame, 0.5)
        # Output should be smaller
        assert out.shape[0] < 480
        assert out.shape[1] < 640
        assert info.resize_scale < 1.0


class TestScaleManagerReverseCenter:
    """Test coordinate reverse mapping."""

    def test_noop_reverse(self):
        sm = ScaleManager()
        info = CropInfo(scale_r=1.0, crop_x=0, crop_y=0,
                        crop_w=640, crop_h=480, resize_scale=1.0)
        pt = np.array([[320.0, 240.0]])
        out = sm.reverse_center(pt, info)
        np.testing.assert_allclose(out, pt)

    def test_crop_reverse(self):
        sm = ScaleManager()
        # Simulating r=2.0: crop center (160,120)-(480,360) then upscale 2x
        info = CropInfo(scale_r=2.0, crop_x=160, crop_y=120,
                        crop_w=320, crop_h=240, resize_scale=2.0)
        # Center of normalised frame (320, 240)
        pt = np.array([[320.0, 240.0]])
        out = sm.reverse_center(pt, info)
        # 320/2 + 160 = 320, 240/2 + 120 = 240
        np.testing.assert_allclose(out, [[320.0, 240.0]])

    def test_downscale_reverse(self):
        sm = ScaleManager()
        info = CropInfo(scale_r=0.5, crop_x=0, crop_y=0,
                        crop_w=640, crop_h=480, resize_scale=0.5)
        # Center of downscaled frame (160, 120)
        pt = np.array([[160.0, 120.0]])
        out = sm.reverse_center(pt, info)
        # 160/0.5 = 320, 120/0.5 = 240
        np.testing.assert_allclose(out, [[320.0, 240.0]])


class TestScaleManagerUpdateFromHomography:
    """Test EMA prior update from homography decomposition."""

    def test_identity_homography_gives_r_near_one(self):
        sm = ScaleManager()
        H = np.eye(3, dtype=np.float64)
        sm.update_from_homography(H, 640, 480)
        assert sm.prior is not None
        assert 0.9 < sm.prior < 1.1

    def test_scaled_homography(self):
        sm = ScaleManager()
        # 2x scale homography
        H = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]], dtype=np.float64)
        sm.update_from_homography(H, 640, 480)
        assert sm.prior is not None
        assert sm.prior > 1.5  # should be ~2.0

    def test_ema_smoothing(self):
        sm = ScaleManager()
        H1 = np.eye(3, dtype=np.float64)
        sm.update_from_homography(H1, 640, 480)
        first_prior = sm.prior

        # Now a 1.5x scale
        H2 = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.0]], dtype=np.float64)
        sm.update_from_homography(H2, 640, 480)
        # EMA: should be between first_prior and 1.5
        assert first_prior < sm.prior < 1.5

    def test_out_of_range_rejected(self):
        sm = ScaleManager()
        # 10x scale — way out of range [0.3, 3.5]
        H = np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 1.0]], dtype=np.float64)
        sm.update_from_homography(H, 640, 480)
        assert sm.prior is None  # rejected


class TestScaleManagerDepthHint:
    """Test depth hint integration."""

    def test_depth_hint_set(self):
        sm = ScaleManager()
        sm.set_depth_hint(0.5, 0.5)  # ratio = 1.0
        assert sm._depth_hint is not None
        assert abs(sm._depth_hint - 1.0) < 0.01

    def test_depth_hint_clipped(self):
        sm = ScaleManager()
        sm.set_depth_hint(100.0, 1.0)  # ratio = 100 → clipped to 3.5
        assert sm._depth_hint == 3.5


class TestScaleManagerReset:
    """Test full reset."""

    def test_reset_clears_all(self):
        sm = ScaleManager()
        sm._prior = 1.5
        sm._depth_hint = 2.0
        sm.reset()
        assert sm.prior is None
        assert sm._depth_hint is None
