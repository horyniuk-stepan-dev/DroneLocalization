import gc
import os
import threading
import time
from contextlib import contextmanager

import torch

from config.config import get_cfg
from src.utils.logging_utils import get_logger, silent_output

# Lazy imports moved to top level as requested
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from lightglue import ALIKED, LightGlue, SuperPoint
except ImportError:
    ALIKED = LightGlue = SuperPoint = None

try:
    from src.models.wrappers.trt_dinov2_wrapper import (
        TensorRTDINOv2Wrapper,
        is_trt_available,
    )
except ImportError:
    TensorRTDINOv2Wrapper = None

    def is_trt_available():
        return False


try:
    from src.models.wrappers.cesp_module import CESP
except ImportError:
    CESP = None

logger = get_logger(__name__)


class ModelManager:
    def __init__(self, config=None, device="cuda"):
        self.config = config or {}

        use_cuda = get_cfg(self.config, "models.use_cuda", True)
        if not use_cuda:
            logger.info("CUDA force disabled in configuration")

        self.device = (
            device if (use_cuda and torch.cuda.is_available() and device == "cuda") else "cpu"
        )
        self.models = {}
        self.model_usage = {}

        # Fix #4: Захист від race condition при паралельному завантаженні моделей (prewarm + main thread)
        self._model_lock = threading.Lock()

        self._pinned_models: set[str] = set()

        # Конфігурація VRAM
        self.max_vram_ratio = get_cfg(self.config, "models.vram_management.max_vram_ratio", 0.8)
        self.default_vram_required = get_cfg(
            self.config, "models.vram_management.default_required_mb", 2000.0
        )

        logger.info(f"ModelManager initialized with device: {self.device}")
        if self.device == "cuda":
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            )

    def get_available_vram_mb(self) -> float:
        if self.device == "cpu":
            return float("inf")
        free_mem, total_mem = torch.cuda.mem_get_info()
        available_mb = free_mem / (1024 * 1024)
        return available_mb

    def _is_torch_compile_supported(self) -> bool:
        """Checks if torch.compile is safe to use in the current environment."""
        if not getattr(torch, "compile", None):
            return False

        use_compile = get_cfg(self.config, "models.performance.torch_compile", False)
        if not use_compile:
            return False

        if self.device == "cpu":
            return False

        # Windows-specific safety check: inductor (default) requires Triton
        if os.name == "nt":
            try:
                # inductor doesn't necessarily need triton imported,
                # but it needs it available in the environment.
                import triton  # noqa: F401

                return True
            except ImportError:
                # If Triton is missing on Windows, torch.compile(backend='inductor')
                # will likely crash with internal errors in dev/nightly PyTorch.
                logger.warning(
                    "torch.compile is requested but Triton is not installed on Windows. "
                    "Compilation disabled to prevent 'inductor' backend crashes."
                )
                return False

        return True

    def pin(self, models: list[str]):
        """Закріплює моделі в пам'яті (запобігає вивантаженню при нестачі VRAM)"""
        with self._model_lock:
            for m in models:
                self._pinned_models.add(m)
            logger.info(f"Pinned models: {self._pinned_models}")

    def unpin_all(self):
        """Знімає закріплення з усіх моделей"""
        with self._model_lock:
            self._pinned_models.clear()
            logger.info("Unpinned all models")

    def _unload_model_unsafe(self, name: str):
        if name in self.models:
            logger.info(f"Unloading model to free VRAM: {name}")
            del self.models[name]
            del self.model_usage[name]
            if self.device != "cpu":
                torch.cuda.empty_cache()
                gc.collect()

    def _ensure_vram_available(self, required_mb: float | None = None):
        if self.device == "cpu":
            return

        req = required_mb if required_mb is not None else self.default_vram_required

        while self.get_available_vram_mb() < req and self.models:
            non_pinned = {k: v for k, v in self.model_usage.items() if k not in self._pinned_models}
            if not non_pinned:
                logger.warning("All models pinned, cannot free VRAM. Risk of OOM.")
                return
            least = min(non_pinned, key=non_pinned.get)
            self._unload_model_unsafe(least)

    def _register_model_usage(self, name: str):
        self.model_usage[name] = time.time()

    def prewarm(self):
        """Centralized model prewarming, usually called at startup in parallel"""
        logger.info("Starting centralized model prewarm sequence...")
        with silent_output():
            self.load_dinov2()
            self.load_aliked()
            self.load_lightglue_aliked()
            self.load_yolo()
        logger.success("Centralized model prewarm complete")

    def load_yolo(self):
        name = "yolo"
        with self._model_lock:
            if name not in self.models:
                model_path = get_cfg(self.config, "models.yolo.model_path", "yolo11n-seg.pt")
                vram_req = get_cfg(self.config, "models.yolo.vram_required_mb", 1200.0)
                use_trt = get_cfg(self.config, "models.performance.use_tensorrt_for_yolo", True)

                logger.info(f"Loading YOLO model: {model_path}...")
                self._ensure_vram_available(vram_req)
                try:
                    if YOLO is None:
                        raise ImportError("ultralytics.YOLO not found")

                    engine_path = str(model_path).replace(".pt", ".engine")
                    import os

                    if use_trt and self.device == "cuda":
                        if os.path.exists(engine_path):
                            logger.info(f"Found YOLO TRT engine: {engine_path}. Loading...")
                            with silent_output():
                                model = YOLO(engine_path, verbose=False)
                        else:
                            logger.info(
                                "YOLO TRT engine not found. Loading PyTorch model for export..."
                            )
                            model = YOLO(model_path, verbose=False)
                            model.to(self.device)
                            logger.info(
                                "Exporting YOLO to TensorRT format (this may take a while)..."
                            )
                            try:
                                # ultralytics automatically places the exported file next to the original
                                exported_path = model.export(
                                    format="engine", half=True, dynamic=True, batch=2,
                                    verbose=False,
                                )
                                logger.success(f"YOLO TRT export complete: {exported_path}")
                                if os.path.exists(exported_path):
                                    with silent_output():
                                        model = YOLO(exported_path, verbose=False)
                                    logger.info("YOLO TensorRT engine loaded successfully.")
                            except Exception as ex:
                                logger.warning(
                                    f"YOLO TRT export failed: {ex}. Falling back to PyTorch."
                                )
                    else:
                        with silent_output():
                            model = YOLO(model_path, verbose=False)
                        model.to(self.device)

                    self.models[name] = model
                    logger.success(f"YOLO model loaded successfully on {self.device}")
                except Exception as e:
                    logger.error(
                        f"Failed to load YOLO model: {e} | "
                        f"model_path={model_path}, device={self.device}, "
                        f"available_vram={self.get_available_vram_mb():.0f} MB. "
                        f"Check that the model file exists and is not corrupted.",
                        exc_info=True,
                    )
                    raise
            self._register_model_usage(name)
            return self.models[name]

    def load_xfeat(self):
        name = "xfeat"
        with self._model_lock:
            if name not in self.models:
                repo = get_cfg(self.config, "models.xfeat.hub_repo", "verlab/accelerated_features")
                model_name = get_cfg(self.config, "models.xfeat.hub_model", "XFeat")
                top_k = get_cfg(self.config, "models.xfeat.top_k", 2048)
                vram_req = get_cfg(self.config, "models.xfeat.vram_required_mb", 300.0)

                logger.info(f"Loading XFeat model ({repo}/{model_name})...")
                self._ensure_vram_available(vram_req)
                try:
                    preset = get_cfg(self.config, "models.xfeat.xfeat_preset", "fast")
                    try:
                        # Attempt to pass quality_preset if supported by User's XFeat fork
                        model = torch.hub.load(
                            repo, model_name, pretrained=True, top_k=top_k, quality_preset=preset
                        )
                    except TypeError:
                        # Fallback to standard XFeat
                        model = torch.hub.load(repo, model_name, pretrained=True, top_k=top_k)

                    # FIX: XFeat hardcodes self.dev='cuda' if available, causing crashes if we move to CPU
                    if hasattr(model, "dev"):
                        model.dev = torch.device(self.device)
                    model = model.eval().to(self.device)
                    self.models[name] = model
                    logger.success(f"XFeat loaded successfully on {self.device} (preset: {preset})")
                except Exception as e:
                    logger.error(
                        f"Failed to load XFeat: {e} | "
                        f"repo={repo}, model={model_name}, device={self.device}, "
                        f"available_vram={self.get_available_vram_mb():.0f} MB. "
                        f"Check internet connection for torch.hub download.",
                        exc_info=True,
                    )
                    raise
            self._register_model_usage(name)
            return self.models[name]

    def load_superpoint(self):
        name = "superpoint"
        with self._model_lock:
            if name not in self.models:
                vram_req = get_cfg(self.config, "models.superpoint.vram_required_mb", 500.0)

                logger.info("Loading SuperPoint model (for LightGlue compatibility)...")
                self._ensure_vram_available(vram_req)
                try:
                    if SuperPoint is None:
                        raise ImportError("lightglue.SuperPoint not found")

                    sp_config = {
                        "nms_radius": get_cfg(self.config, "models.superpoint.nms_radius", 4),
                        "max_num_keypoints": get_cfg(
                            self.config, "models.superpoint.max_keypoints", 4096
                        ),
                    }
                    model = SuperPoint(**sp_config).eval().to(self.device)
                    self.models[name] = model
                    logger.success("SuperPoint model loaded successfully")
                except Exception as e:
                    logger.error(
                        f"Failed to load SuperPoint: {e} | device={self.device}, "
                        f"available_vram={self.get_available_vram_mb():.0f} MB. "
                        f"Check that 'lightglue' package is installed correctly.",
                        exc_info=True,
                    )
                    raise
            self._register_model_usage(name)
            return self.models[name]

    def load_lightglue(self):
        name = "lightglue"
        with self._model_lock:
            if name not in self.models:
                vram_req = get_cfg(self.config, "models.lightglue.vram_required_mb", 1000.0)

                logger.info("Loading LightGlue model...")
                self._ensure_vram_available(vram_req)
                try:
                    if LightGlue is None:
                        raise ImportError("lightglue.LightGlue not found")

                    lg_config = {
                        "depth_confidence": get_cfg(
                            self.config, "models.lightglue.depth_confidence", -1
                        ),
                        "width_confidence": get_cfg(
                            self.config, "models.lightglue.width_confidence", -1
                        ),
                    }
                    model = LightGlue(features="superpoint", **lg_config).eval().to(self.device)
                    self.models[name] = model
                    logger.success("LightGlue model loaded successfully")
                except Exception as e:
                    logger.error(
                        f"Failed to load LightGlue: {e} | device={self.device}, "
                        f"available_vram={self.get_available_vram_mb():.0f} MB. "
                        f"Check that 'lightglue' package is installed and VRAM is sufficient.",
                        exc_info=True,
                    )
                    raise
            self._register_model_usage(name)
            return self.models[name]

    def load_dinov2(self):
        name = "dinov2"
        with self._model_lock:
            if name not in self.models:
                repo = get_cfg(self.config, "models.dinov2.hub_repo", "facebookresearch/dinov2")
                model_name = get_cfg(self.config, "models.dinov2.hub_model", "dinov2_vitl14")
                vram_req = get_cfg(self.config, "models.dinov2.vram_required_mb", 1600.0)

                logger.info(f"Loading DINOv2 ({model_name}) model...")
                self._ensure_vram_available(vram_req)

                # Спроба завантажити TensorRT engine (якщо скомпільований)
                trt_loaded = False
                engine_dir = get_cfg(
                    self.config, "models.engines_cache.engine_cache_dir", "models/engines/"
                )
                try:
                    if TensorRTDINOv2Wrapper is not None and is_trt_available():
                        engine_path = os.path.join(engine_dir, "dinov2_vitl14_fp16.engine")
                        if os.path.exists(engine_path):
                            model = TensorRTDINOv2Wrapper(engine_path)
                            self.models[name] = model
                            trt_loaded = True
                            logger.success(f"DINOv2 TensorRT FP16 engine loaded: {engine_path}")
                except Exception as e:
                    logger.debug(f"TensorRT DINOv2 not available, using PyTorch: {e}")

                # Fallback: стандартний PyTorch hub
                if not trt_loaded:
                    try:
                        model = torch.hub.load(repo, model_name, verbose=False).to(self.device)

                        if self._is_torch_compile_supported():
                            try:
                                # Mode 'default' allows variable batch size gracefully
                                model = torch.compile(model, mode="default")
                                logger.info(
                                    "DINOv2 compiled successfully using torch.compile(mode='default')"
                                )
                            except Exception as e:
                                logger.warning(f"Failed to torch.compile DINOv2: {e}")

                        self.models[name] = model
                        logger.success(f"DINOv2 model {model_name} loaded successfully (PyTorch)")
                    except Exception as e:
                        logger.error(f"Failed to load DINOv2: {e}", exc_info=True)
                        raise
            self._register_model_usage(name)
            return self.models[name]

    def load_aliked(self):
        """Завантажує ALIKED extractor (128-dim, lightglue-compatible)"""
        name = "aliked"
        with self._model_lock:
            if name not in self.models:
                vram_req = get_cfg(self.config, "models.aliked.vram_required_mb", 400.0)
                max_keypoints = get_cfg(self.config, "models.aliked.max_keypoints", 4096)

                logger.info(f"Loading ALIKED model (max_keypoints={max_keypoints})...")
                self._ensure_vram_available(vram_req)
                try:
                    if ALIKED is None:
                        raise ImportError("lightglue.ALIKED not found")

                    model = ALIKED(max_num_keypoints=max_keypoints).eval().to(self.device)

                    if self._is_torch_compile_supported():
                        try:
                            model = torch.compile(model, mode="default")
                            logger.info(
                                "ALIKED compiled successfully using torch.compile(mode='default')"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to torch.compile ALIKED: {e}")

                    self.models[name] = model
                    logger.success(f"ALIKED loaded successfully on {self.device}")
                except Exception as e:
                    logger.error(
                        f"Failed to load ALIKED: {e} | "
                        f"max_keypoints={max_keypoints}, device={self.device}, "
                        f"available_vram={self.get_available_vram_mb():.0f} MB. "
                        f"Check that 'lightglue' package is installed correctly.",
                        exc_info=True,
                    )
                    raise
            self._register_model_usage(name)
            return self.models[name]

    def load_lightglue_aliked(self):
        """Завантажує LightGlue з вагами для ALIKED (128-dim)"""
        name = "lightglue_aliked"
        with self._model_lock:
            if name not in self.models:
                vram_req = get_cfg(self.config, "models.lightglue.vram_required_mb", 1000.0)

                logger.info("Loading LightGlue (ALIKED weights)...")
                self._ensure_vram_available(vram_req)
                try:
                    if LightGlue is None:
                        raise ImportError("lightglue.LightGlue not found")

                    model = (
                        LightGlue(
                            features="aliked",
                            depth_confidence=get_cfg(
                                self.config, "models.lightglue.depth_confidence", -1
                            ),
                            width_confidence=get_cfg(
                                self.config, "models.lightglue.width_confidence", -1
                            ),
                        )
                        .eval()
                        .to(self.device)
                    )
                    self.models[name] = model
                    logger.success("LightGlue (ALIKED) loaded successfully")
                except Exception as e:
                    logger.error(
                        f"Failed to load LightGlue (ALIKED): {e} | device={self.device}, "
                        f"available_vram={self.get_available_vram_mb():.0f} MB",
                        exc_info=True,
                    )
                    raise
            self._register_model_usage(name)
            return self.models[name]

    def load_cesp(self):
        """Завантажує CESP модуль для покращення DINOv2 global descriptors"""
        name = "cesp"
        with self._model_lock:
            if name not in self.models:
                logger.info("Loading CESP module...")
                try:
                    if CESP is None:
                        raise ImportError("CESP not found")

                    scales = get_cfg(self.config, "models.cesp.scales", [1, 2, 4])
                    cesp = CESP(dim=1024, scales=tuple(scales))

                    # Завантаження pretrained ваг (якщо є)
                    weights_path = get_cfg(self.config, "models.cesp.weights_path", None)
                    if weights_path:
                        cesp.load_state_dict(torch.load(weights_path, map_location=self.device))
                        logger.success(f"CESP pretrained weights loaded from {weights_path}")
                    else:
                        logger.warning("CESP initialized WITHOUT pretrained weights (random init)")

                    cesp = cesp.eval().to(self.device)
                    self.models[name] = cesp
                    logger.success("CESP module loaded")
                except Exception as e:
                    logger.error(
                        f"Failed to load CESP: {e} | "
                        f"weights_path={weights_path}, device={self.device}. "
                        f"Check that the weights file exists and is compatible.",
                        exc_info=True,
                    )
                    raise
            self._register_model_usage(name)
            return self.models[name]

    def unload_model(self, model_name: str):
        with self._model_lock:
            self._unload_model_unsafe(model_name)

    @contextmanager
    def inference_context(self):
        try:
            with torch.no_grad():
                yield
        finally:
            if self.device != "cpu":
                torch.cuda.empty_cache()
