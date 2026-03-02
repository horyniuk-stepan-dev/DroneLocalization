import torch
import gc
import time
from contextlib import contextmanager
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ModelManager:
    def __init__(self, config=None, device='cuda'):
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.models = {}
        self.model_usage = {}
        self.max_vram_ratio = 0.8

        logger.info(f"ModelManager initialized with device: {self.device}")
        if self.device == 'cuda':
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    def get_available_vram_mb(self) -> float:
        if self.device == 'cpu':
            return float('inf')
        free_mem, total_mem = torch.cuda.mem_get_info()
        available_mb = free_mem / (1024 * 1024)
        return available_mb

    def _ensure_vram_available(self, required_mb: float = 2000.0):
        if self.device == 'cpu':
            return
        while self.get_available_vram_mb() < required_mb and self.models:
            least_used = min(self.model_usage.items(), key=lambda x: x[1])[0]
            logger.warning(f"VRAM insufficient. Unloading least used model: {least_used}")
            self.unload_model(least_used)

    def _register_model_usage(self, name: str):
        self.model_usage[name] = time.time()

    def load_yolo(self):
        name = 'yolo'
        if name not in self.models:
            logger.info(f"Loading YOLOv11n-seg model...")
            self._ensure_vram_available(300.0) # YOLO11n дуже легка
            try:
                from ultralytics import YOLO
                # Ultralytics автоматично завантажить архітектуру YOLOv11
                model = YOLO('yolo11n-seg.pt')
                model.to(self.device)
                self.models[name] = model
                logger.success(f"YOLOv11n-seg model loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_xfeat(self):
        name = 'xfeat'
        if name not in self.models:
            logger.info(f"Loading XFeat model...")
            self._ensure_vram_available(300.0)
            try:
                # Завантаження XFeat з PyTorch Hub
                model = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=2048)
                model = model.eval().to(self.device)
                self.models[name] = model
                logger.success(f"XFeat loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load XFeat: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_superpoint(self):
        name = 'superpoint'
        if name not in self.models:
            logger.info(f"Loading SuperPoint model (for LightGlue compatibility)...")
            self._ensure_vram_available(500.0)
            try:
                from lightglue import SuperPoint
                sp_config = {
                    'nms_radius': self.config.get('superpoint', {}).get('nms_radius', 4),
                    'max_num_keypoints': self.config.get('superpoint', {}).get('max_keypoints', 2048),
                }
                model = SuperPoint(**sp_config).eval().to(self.device)
                self.models[name] = model
                logger.success(f"SuperPoint model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load SuperPoint model: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_lightglue(self):
        name = 'lightglue'
        if name not in self.models:
            logger.info(f"Loading LightGlue model...")
            self._ensure_vram_available(1000.0)
            try:
                from lightglue import LightGlue
                lg_config = {
                    'depth_confidence': self.config.get('lightglue', {}).get('depth_confidence', -1),
                    'width_confidence': self.config.get('lightglue', {}).get('width_confidence', -1),
                }
                model = LightGlue(features='superpoint', **lg_config).eval().to(self.device)
                self.models[name] = model
                logger.success(f"LightGlue model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load LightGlue: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_dinov2(self):
        name = 'dinov2'
        if name not in self.models:
            logger.info(f"Loading DINOv2 (vits14) model...")
            self._ensure_vram_available(500.0)
            try:
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                model = model.eval().to(self.device)
                self.models[name] = model
                logger.success(f"DINOv2 model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load DINOv2: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def unload_model(self, model_name: str):
        if model_name in self.models:
            logger.info(f"Unloading model: {model_name}")
            del self.models[model_name]
            del self.model_usage[model_name]
            if self.device != 'cpu':
                torch.cuda.empty_cache()
                gc.collect()

    @contextmanager
    def inference_context(self):
        try:
            with torch.no_grad():
                yield
        finally:
            if self.device != 'cpu':
                torch.cuda.empty_cache()