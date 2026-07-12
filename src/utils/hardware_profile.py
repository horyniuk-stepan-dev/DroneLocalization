"""Hardware auto-detection and compute auto-tuning.

Probes GPU (VRAM, compute capability, architecture) and CPU (cores, RAM)
at startup, classifies the system into performance tiers, and returns
optimal overrides for all compute-sensitive config fields.

Usage::

    from src.utils.hardware_profile import HardwareProfile

    profile = HardwareProfile.detect()
    profile.log_summary()
    overrides = profile.auto_tune(current_config_dict)
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Any

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Tier thresholds ──────────────────────────────────────────────────────────
# Tier is determined by VRAM first, then CPU cores as a secondary factor.
_TIER_VRAM_THRESHOLDS = {
    # (min_vram_gb, tier_name)
    "ultra": 16.0,
    "high": 10.0,
    "mid": 6.0,
    "low": 0.0,  # fallback
}


def _classify_tier(vram_gb: float, cpu_cores: int) -> str:
    """Classify into low / mid / high / ultra based on VRAM and CPU cores."""
    if vram_gb >= _TIER_VRAM_THRESHOLDS["ultra"]:
        return "ultra"
    if vram_gb >= _TIER_VRAM_THRESHOLDS["high"]:
        return "high"
    if vram_gb >= _TIER_VRAM_THRESHOLDS["mid"]:
        # Demote to low if CPU is also very weak (≤2 cores)
        return "mid" if cpu_cores > 2 else "low"
    return "low"


# ── Database-interchangeability guard ────────────────────────────────────────
# auto_tune() may change ONLY these keys. Every one affects SPEED (or, for
# fp16/torch_compile, tiny numeric wobble), never the STRUCTURE or content-TYPE
# of a database. Any dot-path NOT in this set is refused by _propose() below
# (fail-closed) — so no hardware difference can ever make two machines build
# non-interchangeable databases. Structure-defining keys (local_extractor,
# global_descriptor.backend, vlad.*, database.max_keypoints_stored,
# keypoint_video_scale, frame_step, sift_max_keypoints, store_sift_features)
# are deliberately ABSENT and must be set explicitly in user_config.json.
TUNABLE_KEYS: frozenset[str] = frozenset({
    "models.vram_management.max_vram_ratio",
    "models.performance.torch_compile",
    "models.performance.fp16_enabled",
    "models.performance.propagation_max_workers",
    "database.yolo_batch_size",
    "database.prefetch_queue_size",
    "database.decode_batch_size",
})


@dataclass
class GPUInfo:
    """Detected GPU properties."""
    available: bool = False
    name: str = "N/A"
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    compute_capability: tuple[int, int] = (0, 0)
    # SM 8.0+ = Ampere (A100/RTX30xx), SM 8.6+ (RTX30xx consumer), SM 8.9 (RTX40xx)
    is_ampere_plus: bool = False
    multi_gpu_count: int = 0
    driver_version: str = "N/A"


@dataclass
class CPUInfo:
    """Detected CPU properties."""
    physical_cores: int = 1
    logical_threads: int = 1
    ram_total_gb: float = 0.0
    architecture: str = "unknown"


@dataclass
class HardwareProfile:
    """Full hardware profile with tier classification and auto-tune capability."""
    gpu: GPUInfo = field(default_factory=GPUInfo)
    cpu: CPUInfo = field(default_factory=CPUInfo)
    tier: str = "low"
    os_name: str = "unknown"

    @classmethod
    def detect(cls) -> "HardwareProfile":
        """Probe the current system and return a populated HardwareProfile."""
        profile = cls()
        profile.os_name = platform.system()

        # ── CPU Detection ────────────────────────────────────────────────────
        profile.cpu = cls._detect_cpu()

        # ── GPU Detection ────────────────────────────────────────────────────
        profile.gpu = cls._detect_gpu()

        # ── Tier Classification ──────────────────────────────────────────────
        vram = profile.gpu.vram_total_gb if profile.gpu.available else 0.0
        profile.tier = _classify_tier(vram, profile.cpu.physical_cores)

        return profile

    @staticmethod
    def _detect_cpu() -> CPUInfo:
        """Detect CPU core count and RAM."""
        info = CPUInfo()
        info.architecture = platform.machine()

        # Physical cores (fallback to logical if unavailable)
        try:
            import psutil
            info.physical_cores = psutil.cpu_count(logical=False) or 1
            info.logical_threads = psutil.cpu_count(logical=True) or info.physical_cores
            info.ram_total_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            # psutil not available — use os.cpu_count (returns logical threads)
            logical = os.cpu_count() or 1
            info.logical_threads = logical
            # Heuristic: assume ~half are physical on x86 with HT
            info.physical_cores = max(1, logical // 2)
            # RAM fallback for Windows
            if platform.system() == "Windows":
                try:
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    c_ulong = ctypes.c_ulonglong
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", c_ulong),
                            ("ullAvailPhys", c_ulong),
                            ("ullTotalPageFile", c_ulong),
                            ("ullAvailPageFile", c_ulong),
                            ("ullTotalVirtual", c_ulong),
                            ("ullAvailVirtual", c_ulong),
                            ("ullAvailExtendedVirtual", c_ulong),
                        ]
                    stat = MEMORYSTATUSEX()
                    stat.dwLength = ctypes.sizeof(stat)
                    kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                    info.ram_total_gb = stat.ullTotalPhys / (1024 ** 3)
                except Exception:
                    info.ram_total_gb = 0.0
            else:
                try:
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if line.startswith("MemTotal"):
                                info.ram_total_gb = int(line.split()[1]) / (1024 ** 2)
                                break
                except Exception:
                    info.ram_total_gb = 0.0

        return info

    @staticmethod
    def _detect_gpu() -> GPUInfo:
        """Detect GPU via PyTorch CUDA."""
        info = GPUInfo()
        try:
            import torch
            if not torch.cuda.is_available():
                return info

            info.available = True
            info.multi_gpu_count = torch.cuda.device_count()
            info.name = torch.cuda.get_device_name(0)

            props = torch.cuda.get_device_properties(0)
            info.vram_total_gb = props.total_memory / (1024 ** 3)
            info.compute_capability = (props.major, props.minor)
            # Ampere = SM 8.0+
            info.is_ampere_plus = props.major >= 8

            free_mem, _ = torch.cuda.mem_get_info(0)
            info.vram_free_gb = free_mem / (1024 ** 3)

            # Driver version (NVML)
            try:
                info.driver_version = torch._C._cuda_getDriverVersion()
                # Returns int like 12040 → "12.4"
                if isinstance(info.driver_version, int):
                    major = info.driver_version // 1000
                    minor = (info.driver_version % 1000) // 10
                    info.driver_version = f"{major}.{minor}"
            except Exception:
                info.driver_version = "N/A"

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")

        return info

    def log_summary(self) -> None:
        """Log the full hardware profile to the application logger."""
        sep = "=" * 70
        logger.info(sep)
        logger.info("HARDWARE PROFILE")
        logger.info(sep)

        if self.gpu.available:
            sm = f"SM {self.gpu.compute_capability[0]}.{self.gpu.compute_capability[1]}"
            ampere = " (Ampere+)" if self.gpu.is_ampere_plus else ""
            logger.info(
                f"GPU:    {self.gpu.name} | "
                f"{self.gpu.vram_total_gb:.1f} GB VRAM "
                f"({self.gpu.vram_free_gb:.1f} GB free) | "
                f"{sm}{ampere}"
            )
            if self.gpu.multi_gpu_count > 1:
                logger.info(f"        Multi-GPU: {self.gpu.multi_gpu_count} devices detected")
        else:
            logger.warning("GPU:    No CUDA GPU detected — running on CPU only")

        logger.info(
            f"CPU:    {self.cpu.physical_cores} cores / "
            f"{self.cpu.logical_threads} threads | "
            f"{self.cpu.ram_total_gb:.1f} GB RAM"
        )
        logger.info(f"Tier:   {self.tier.upper()}")
        logger.info(sep)

    def auto_tune(self, current_config: dict[str, Any]) -> dict[str, tuple[Any, Any, str]]:
        """Compute optimal config overrides based on detected hardware.

        Returns a dict of ``{dot_path: (old_value, new_value, reason)}``
        for every setting that should be changed.  Values the user has
        explicitly customized (non-default) are **never** overwritten.

        The caller is responsible for applying the overrides to the live
        config objects.
        """
        overrides: dict[str, tuple[Any, Any, str]] = {}
        tier = self.tier
        gpu = self.gpu
        cpu = self.cpu

        # ── Helper: only override if current value equals the default ────────
        def _propose(path: str, default_val: Any, new_val: Any, reason: str):
            """Propose an override only if the current config holds the default value."""
            # Fail-closed interchangeability guard: refuse any non-tunable key so
            # hardware can never alter a database's structure/content-type.
            if path not in TUNABLE_KEYS:
                logger.error(
                    "auto_tune BLOCKED non-tunable key %r — only speed keys may be "
                    "auto-tuned; structure-defining keys stay hardware-independent "
                    "so databases remain interchangeable." % path
                )
                return
            # Navigate dot-path in the config dict
            keys = path.split(".")
            current = current_config
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    current = default_val  # not found → treat as default
                    break
            if current == default_val and new_val != default_val:
                overrides[path] = (default_val, new_val, reason)

        # ══════════════════════════════════════════════════════════════════════
        # GPU-side tuning
        # ══════════════════════════════════════════════════════════════════════

        if gpu.available:
            # VRAM ratio — higher on beefier cards
            vram_ratios = {"low": 0.75, "mid": 0.8, "high": 0.85, "ultra": 0.9}
            _propose(
                "models.vram_management.max_vram_ratio",
                0.8,
                vram_ratios[tier],
                f"{tier}-tier GPU: safe to use {vram_ratios[tier]*100:.0f}% VRAM",
            )

            # YOLO batch size — more VRAM → bigger batches
            yolo_batches = {"low": 1, "mid": 2, "high": 4, "ultra": 8}
            _propose(
                "database.yolo_batch_size",
                1,
                yolo_batches[tier],
                f"{tier}-tier GPU: YOLO micro-batch {yolo_batches[tier]}",
            )

            # torch.compile — only on high+ tier (requires Triton on Windows)
            if tier in ("high", "ultra"):
                _propose(
                    "models.performance.torch_compile",
                    False,
                    True,
                    f"{tier}-tier GPU: torch.compile may improve throughput",
                )

            # FP16 — force on low tier to save VRAM
            if tier == "low":
                _propose(
                    "models.performance.fp16_enabled",
                    True,
                    True,
                    "low-tier GPU: FP16 required to fit models in VRAM",
                )

            # NOTE: max_keypoints is intentionally NOT auto-tuned.
            # It defines the HDF5 dataset shape (num_frames, max_kps, ...) and
            # changing it per-system would make databases non-interchangeable.
            # Users who want more keypoints should set it explicitly in user_config.json.

        # ══════════════════════════════════════════════════════════════════════
        # CPU-side tuning
        # ══════════════════════════════════════════════════════════════════════

        # Prefetch queue — scale with thread count and RAM
        prefetch_sizes = {"low": 16, "mid": 32, "high": 64, "ultra": 96}
        # Use tier but also floor at cpu-derived value
        cpu_prefetch = max(32, cpu.logical_threads * 4)
        target_prefetch = max(prefetch_sizes.get(tier, 32), cpu_prefetch)
        # Cap at reasonable maximum
        target_prefetch = min(target_prefetch, 128)
        _propose(
            "database.prefetch_queue_size",
            32,
            target_prefetch,
            f"{cpu.logical_threads} threads → prefetch queue {target_prefetch}",
        )

        # Decode batch size — scale with CPU+GPU capability
        decode_batches = {"low": 16, "mid": 32, "high": 64, "ultra": 96}
        _propose(
            "database.decode_batch_size",
            32,
            decode_batches[tier],
            f"{tier}-tier: decode batch {decode_batches[tier]}",
        )

        # Propagation workers — match physical cores (capped)
        prop_workers = min(cpu.physical_cores, 8)
        _propose(
            "models.performance.propagation_max_workers",
            4,
            prop_workers,
            f"{cpu.physical_cores} physical cores → {prop_workers} propagation workers",
        )

        return overrides

    def apply_overrides(
        self, config_dict: dict[str, Any], overrides: dict[str, tuple[Any, Any, str]]
    ) -> None:
        """Apply computed overrides to a mutable config dictionary in-place."""
        for dot_path, (_, new_val, _) in overrides.items():
            keys = dot_path.split(".")
            target = config_dict
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = new_val

    def apply_torch_backends(self) -> None:
        """Configure PyTorch global backend settings based on hardware.

        Should be called once at startup, after detection. Sets:
        - ``torch.backends.cudnn.benchmark`` for CNN workloads
        - TF32 matmul/convolution on Ampere+ GPUs
        - ``torch.set_num_threads`` for CPU parallelism
        - ``cv2.setNumThreads`` for OpenCV parallelism
        """
        # ── CPU thread tuning ────────────────────────────────────────────────
        try:
            import torch
            physical = self.cpu.physical_cores
            torch.set_num_threads(physical)
            logger.info(f"torch.set_num_threads({physical})")
        except Exception as e:
            logger.debug(f"Could not set torch num_threads: {e}")

        try:
            import cv2
            physical = self.cpu.physical_cores
            cv2.setNumThreads(physical)
            logger.info(f"cv2.setNumThreads({physical})")
        except Exception as e:
            logger.debug(f"Could not set cv2 num_threads: {e}")

        if not self.gpu.available:
            return

        try:
            import torch

            # cuDNN benchmark: always beneficial for repeated same-size convolutions
            torch.backends.cudnn.benchmark = True
            logger.info("cudnn.benchmark = True")

            # TF32: ~2× matmul throughput on Ampere+ with negligible precision loss
            # for inference workloads (DINOv2, ALIKED, LightGlue, YOLO)
            if self.gpu.is_ampere_plus:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info(
                    "TF32 matmul ENABLED (Ampere+ GPU detected: "
                    f"SM {self.gpu.compute_capability[0]}.{self.gpu.compute_capability[1]})"
                )
            else:
                logger.info(
                    "TF32 matmul not available "
                    f"(SM {self.gpu.compute_capability[0]}.{self.gpu.compute_capability[1]} < 8.0)"
                )

        except Exception as e:
            logger.warning(f"Failed to configure torch backends: {e}")

    def log_overrides(self, overrides: dict[str, tuple[Any, Any, str]]) -> None:
        """Pretty-print applied overrides to the log."""
        if not overrides:
            logger.info("Auto-tune: no overrides needed (all settings already optimal or customized)")
            return

        logger.info("Auto-tune applied:")
        items = list(overrides.items())
        for i, (path, (old, new, reason)) in enumerate(items):
            connector = "└─" if i == len(items) - 1 else "├─"
            short_path = path.split(".")[-1]
            logger.info(f"  {connector} {short_path}: {old} → {new} ({reason})")
