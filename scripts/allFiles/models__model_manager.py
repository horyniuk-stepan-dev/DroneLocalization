import gc
import time
import torch
from contextlib import contextmanager
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ModelManager:
    def __init__(self, config=None, device='cuda'):
        self.config = config or {}
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.models = {}
        self.model_usage = {}
        self.max_vram_ratio = 0.8

        logger.info(f"ModelManager initialized on device: {self.device}")
        if self.device == 'cuda':
            props = torch.cuda.get_device_properties(0)
            logger.info(f"CUDA: {props.name}, VRAM: {props.total_memory / 1024**3:.2f} GB")

    def get_available_vram_mb(self) -> float:
        if self.device == 'cpu':
            return float('inf')
        free_mem, _ = torch.cuda.mem_get_info()
        return free_mem / (1024 * 1024)

    def _ensure_vram_available(self, required_mb: float = 2000.0):
        if self.device == 'cpu':
            return
        while self.get_available_vram_mb() < required_mb and self.models:
            least_used = min(self.model_usage, key=self.model_usage.get)
            logger.warning(f"VRAM low — unloading: {least_used}")
            self.unload_model(least_used)

    def _register_model_usage(self, name: str):
        self.model_usage[name] = time.time()

    def load_yolo(self):
        name = 'yolo'
        if name not in self.models:
            self._ensure_vram_available(2000.0)
            try:
                from ultralytics import YOLO
                self.models[name] = YOLO('yolo11n-seg.pt')
                logger.success(f"YOLO loaded (inference device: {self.device})")
            except Exception as e:
                logger.error(f"Failed to load YOLO: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_superpoint(self):
        """Loads XFeat as SuperPoint-compatible feature extractor for LightGlue."""
        name = 'superpoint'
        if name not in self.models:
            self._ensure_vram_available(1000.0)
            try:
                xfeat = torch.hub.load(
                    'verlab/accelerated_features', 'XFeat',
                    pretrained=True, top_k=2048
                ).eval().to(self.device)
                self.models[name] = XFeatAdapter(xfeat)
                logger.success(f"XFeat (SuperPoint adapter) loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load XFeat: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_lightglue(self):
        name = 'lightglue'
        if name not in self.models:
            self._ensure_vram_available(1500.0)
            try:
                from lightglue import LightGlue
                lg_cfg = self.config.get('lightglue', {})
                lg_config = {
                    'depth_confidence': lg_cfg.get('depth_confidence', -1),
                    'width_confidence': lg_cfg.get('width_confidence', -1),
                    'filter_threshold': lg_cfg.get('filter_threshold', 0.1),
                }
                self.models[name] = LightGlue(
                    features='superpoint', **lg_config
                ).eval().to(self.device)
                logger.success(f"LightGlue loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load LightGlue: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_dinov2(self):
        name = 'dinov2'
        if name not in self.models:
            self._ensure_vram_available(500.0)
            try:
                model = torch.hub.load(
                    'facebookresearch/dinov2', 'dinov2_vits14'
                ).eval().to(self.device)
                self.models[name] = model
                logger.success(f"DINOv2 loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load DINOv2: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def unload_model(self, model_name: str):
        if model_name not in self.models:
            logger.warning(f"Attempted to unload non-existent model: {model_name}")
            return
        del self.models[model_name]
        del self.model_usage[model_name]
        if self.device != 'cpu':
            torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Model unloaded: {model_name}")

    @contextmanager
    def inference_context(self):
        with torch.no_grad():
            yield


class XFeatAdapter:
    """Wraps XFeat to match SuperPoint output format expected by LightGlue."""

    def __init__(self, model):
        self.model = model

    def __call__(self, data):
        img = data['image']
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        out = self.model.detectAndCompute(img, top_k=2048)[0]
        return {
            'keypoints': out['keypoints'].unsqueeze(0),
            'keypoint_scores': out['scores'].unsqueeze(0),
            'descriptors': out['descriptors'].unsqueeze(0),
        }

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model = self.model.to(device)
        return self
