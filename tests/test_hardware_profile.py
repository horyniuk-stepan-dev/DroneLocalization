"""Tests for the hardware auto-detection and auto-tuning system."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.utils.hardware_profile import (
    CPUInfo,
    GPUInfo,
    HardwareProfile,
    _classify_tier,
)


# ── Tier classification ──────────────────────────────────────────────────────


class TestClassifyTier:
    def test_ultra_tier(self):
        assert _classify_tier(24.0, 16) == "ultra"
        assert _classify_tier(16.0, 8) == "ultra"

    def test_high_tier(self):
        assert _classify_tier(12.0, 8) == "high"
        assert _classify_tier(10.0, 4) == "high"

    def test_mid_tier(self):
        assert _classify_tier(8.0, 6) == "mid"
        assert _classify_tier(6.0, 4) == "mid"

    def test_mid_demoted_to_low_on_weak_cpu(self):
        # 8 GB VRAM but only 2 CPU cores → demoted to low
        assert _classify_tier(8.0, 2) == "low"

    def test_low_tier(self):
        assert _classify_tier(4.0, 4) == "low"
        assert _classify_tier(0.0, 1) == "low"

    def test_cpu_only(self):
        assert _classify_tier(0.0, 8) == "low"


# ── Auto-tune logic ─────────────────────────────────────────────────────────


def _make_profile(vram_gb: float, cpu_cores: int, is_ampere: bool = False) -> HardwareProfile:
    """Create a HardwareProfile with mocked hardware values."""
    gpu = GPUInfo(
        available=vram_gb > 0,
        name="Test GPU",
        vram_total_gb=vram_gb,
        vram_free_gb=vram_gb * 0.9,
        compute_capability=(8, 6) if is_ampere else (7, 5),
        is_ampere_plus=is_ampere,
        multi_gpu_count=1 if vram_gb > 0 else 0,
    )
    cpu = CPUInfo(
        physical_cores=cpu_cores,
        logical_threads=cpu_cores * 2,
        ram_total_gb=32.0,
    )
    tier = _classify_tier(vram_gb, cpu_cores)
    return HardwareProfile(gpu=gpu, cpu=cpu, tier=tier)


def _default_config() -> dict:
    """Minimal config dict with default values that the auto-tuner checks against."""
    return {
        "models": {
            "vram_management": {"max_vram_ratio": 0.8},
            "performance": {
                "auto_tune": True,
                "propagation_max_workers": 4,
                "torch_compile": False,
                "fp16_enabled": True,
            },
            "aliked": {"max_keypoints": 4096},
            "rdd": {"max_keypoints": 4096},
            "superpoint": {"max_keypoints": 4096},
            "xfeat": {"max_keypoints": 4096},
        },
        "database": {
            "prefetch_queue_size": 32,
            "decode_batch_size": 32,
            "yolo_batch_size": 1,
        },
    }


class TestAutoTune:
    def test_ultra_tier_increases_batch_sizes(self):
        profile = _make_profile(24.0, 16, is_ampere=True)
        overrides = profile.auto_tune(_default_config())

        # YOLO batch should be increased
        assert "database.yolo_batch_size" in overrides
        _, new_val, _ = overrides["database.yolo_batch_size"]
        assert new_val == 8

    def test_ultra_tier_increases_vram_ratio(self):
        profile = _make_profile(24.0, 16)
        overrides = profile.auto_tune(_default_config())

        assert "models.vram_management.max_vram_ratio" in overrides
        _, new_val, _ = overrides["models.vram_management.max_vram_ratio"]
        assert new_val == 0.9

    def test_high_tier_enables_torch_compile(self):
        profile = _make_profile(12.0, 8)
        overrides = profile.auto_tune(_default_config())

        assert "models.performance.torch_compile" in overrides
        _, new_val, _ = overrides["models.performance.torch_compile"]
        assert new_val is True

    def test_low_tier_reduces_batch_sizes(self):
        profile = _make_profile(4.0, 4)
        overrides = profile.auto_tune(_default_config())

        # Prefetch should be adjusted
        if "database.decode_batch_size" in overrides:
            _, new_val, _ = overrides["database.decode_batch_size"]
            assert new_val <= 32  # low tier should not increase

    def test_user_customized_values_not_overwritten(self):
        """If user has set a value different from default, auto-tune should NOT touch it."""
        config = _default_config()
        config["database"]["yolo_batch_size"] = 16  # user-customized

        profile = _make_profile(24.0, 16)
        overrides = profile.auto_tune(config)

        # yolo_batch_size should NOT appear in overrides (user already customized it)
        assert "database.yolo_batch_size" not in overrides

    def test_cpu_only_no_gpu_overrides(self):
        profile = _make_profile(0.0, 8)
        overrides = profile.auto_tune(_default_config())

        # Should not contain GPU-specific overrides
        assert "models.vram_management.max_vram_ratio" not in overrides
        assert "database.yolo_batch_size" not in overrides

    def test_apply_overrides_modifies_dict(self):
        config = _default_config()
        profile = _make_profile(24.0, 16)
        overrides = profile.auto_tune(config)

        profile.apply_overrides(config, overrides)

        assert config["database"]["yolo_batch_size"] == 8
        assert config["models"]["vram_management"]["max_vram_ratio"] == 0.9

    def test_mid_tier_keypoints_unchanged(self):
        """Mid tier has default 4096 keypoints — no override needed."""
        profile = _make_profile(8.0, 6)
        overrides = profile.auto_tune(_default_config())

        # mid tier keypoints = 4096 = default → no override
        assert "models.aliked.max_keypoints" not in overrides

    def test_ultra_tier_keypoints_unchanged(self):
        """Interchangeability: the keypoint budget defines the HDF5 dataset shape,
        so it must NOT scale with hardware — an ultra-tier machine must build the
        same database schema as a low-tier one. (Formerly
        test_ultra_tier_keypoints_increased, which asserted the old
        hardware-dependent behavior that made databases non-interchangeable.)"""
        profile = _make_profile(24.0, 16)
        overrides = profile.auto_tune(_default_config())

        assert "models.aliked.max_keypoints" not in overrides

    def test_propagation_workers_scaled_to_cores(self):
        profile = _make_profile(12.0, 12)
        overrides = profile.auto_tune(_default_config())

        assert "models.performance.propagation_max_workers" in overrides
        _, new_val, _ = overrides["models.performance.propagation_max_workers"]
        assert new_val == 8  # capped at 8


# ── HardwareProfile.detect() integration ────────────────────────────────────


class TestDetect:
    @patch("src.utils.hardware_profile.HardwareProfile._detect_gpu")
    @patch("src.utils.hardware_profile.HardwareProfile._detect_cpu")
    def test_detect_returns_valid_profile(self, mock_cpu, mock_gpu):
        mock_cpu.return_value = CPUInfo(physical_cores=8, logical_threads=16, ram_total_gb=32.0)
        mock_gpu.return_value = GPUInfo(
            available=True,
            name="Mock GPU",
            vram_total_gb=12.0,
            vram_free_gb=11.0,
            compute_capability=(8, 6),
            is_ampere_plus=True,
            multi_gpu_count=1,
        )

        profile = HardwareProfile.detect()
        assert profile.tier == "high"
        assert profile.gpu.is_ampere_plus is True
        assert profile.cpu.physical_cores == 8

    @patch("src.utils.hardware_profile.HardwareProfile._detect_gpu")
    @patch("src.utils.hardware_profile.HardwareProfile._detect_cpu")
    def test_detect_cpu_only(self, mock_cpu, mock_gpu):
        mock_cpu.return_value = CPUInfo(physical_cores=4, logical_threads=8, ram_total_gb=16.0)
        mock_gpu.return_value = GPUInfo()  # defaults: available=False

        profile = HardwareProfile.detect()
        assert profile.tier == "low"
        assert profile.gpu.available is False


# ── apply_torch_backends ─────────────────────────────────────────────────────


class TestApplyTorchBackends:
    @patch("torch.set_num_threads")
    @patch("cv2.setNumThreads")
    def test_sets_thread_counts(self, mock_cv2, mock_torch):
        profile = _make_profile(0.0, 8)
        profile.apply_torch_backends()

        mock_torch.assert_called_once_with(8)
        mock_cv2.assert_called_once_with(8)
