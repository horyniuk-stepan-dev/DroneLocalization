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
        logger.debug(f"Available VRAM: {available_mb:.2f} MB / {total_mem / 1024 ** 3:.2f} GB total")
        return available_mb

    def _ensure_vram_available(self, required_mb: float = 2000.0):
        if self.device == 'cpu':
            return

        current_available = self.get_available_vram_mb()
        logger.debug(f"Ensuring {required_mb:.2f} MB VRAM. Currently available: {current_available:.2f} MB")

        while self.get_available_vram_mb() < required_mb and self.models:
            least_used = min(self.model_usage.items(), key=lambda x: x[1])[0]
            logger.warning(f"VRAM insufficient. Unloading least used model: {least_used}")
            self.unload_model(least_used)

    def _register_model_usage(self, name: str):
        self.model_usage[name] = time.time()
        logger.debug(f"Registered model usage: {name}")

    def load_yolo(self):
        name = 'yolo'
        if name not in self.models:
            logger.info(f"Loading YOLO model...")
            self._ensure_vram_available(2000.0)

            try:
                from ultralytics import YOLO
                model = YOLO('yolov8x-seg.pt')
                model.to(self.device)
                self.models[name] = model
                logger.success(f"YOLO model loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
                raise
        else:
            logger.debug(f"YOLO model already loaded, reusing cached instance")

        self._register_model_usage(name)
        return self.models[name]

    def load_superpoint(self):
        name = 'superpoint'
        if name not in self.models:
            logger.info(f"Loading SuperPoint model...")
            self._ensure_vram_available(1000.0)

            try:
                # Використовуємо вбудований SuperPoint з LightGlue замість hloc
                from lightglue import SuperPoint

                # Адаптуємо назви параметрів конфігурації під стандарт LightGlue
                sp_config = {
                    'nms_radius': self.config.get('superpoint', {}).get('nms_radius', 4),
                    'detection_threshold': self.config.get('superpoint', {}).get('keypoint_threshold', 0.005),
                    'max_num_keypoints': self.config.get('superpoint', {}).get('max_keypoints', 2048),
                }

                logger.debug(f"SuperPoint config: {sp_config}")

                # Завантажуємо модель
                model = SuperPoint(**sp_config).eval().to(self.device)
                self.models[name] = model

                logger.success(f"SuperPoint model loaded successfully on {self.device}")
                logger.info(f"SuperPoint max keypoints: {sp_config['max_num_keypoints']}")

            except Exception as e:
                logger.error(f"Failed to load SuperPoint model: {e}", exc_info=True)
                raise
        else:
            logger.debug(f"SuperPoint model already loaded, reusing cached instance")

        self._register_model_usage(name)
        return self.models[name]

    def load_netvlad(self):
        name = 'netvlad'
        if name not in self.models:
            logger.info(f"Loading NetVLAD model...")
            self._ensure_vram_available(2000.0)

            try:
                from hloc.extractors import netvlad

                # NetVLAD configuration
                nv_config = self.config.get('netvlad', {})

                logger.debug(f"NetVLAD config: {nv_config}")

                # Load model
                model = netvlad.NetVLAD(nv_config).eval().to(self.device)
                self.models[name] = model

                logger.success(f"NetVLAD model loaded successfully on {self.device}")

            except Exception as e:
                logger.error(f"Failed to load NetVLAD model: {e}", exc_info=True)
                raise
        else:
            logger.debug(f"NetVLAD model already loaded, reusing cached instance")

        self._register_model_usage(name)
        return self.models[name]

    def load_lightglue(self):
        name = 'lightglue'
        if name not in self.models:
            logger.info(f"Loading LightGlue model...")
            self._ensure_vram_available(1500.0)

            try:
                from lightglue import LightGlue, SuperPoint as LG_SuperPoint

                # LightGlue configuration
                lg_config = {
                    'depth_confidence': self.config.get('lightglue', {}).get('depth_confidence', -1),
                    'width_confidence': self.config.get('lightglue', {}).get('width_confidence', -1),
                    'filter_threshold': self.config.get('lightglue', {}).get('filter_threshold', 0.1),
                }

                logger.debug(f"LightGlue config: {lg_config}")

                # Load model (configure for SuperPoint features)
                model = LightGlue(features='superpoint', **lg_config).eval().to(self.device)
                self.models[name] = model

                logger.success(f"LightGlue model loaded successfully on {self.device}")

            except Exception as e:
                logger.error(f"Failed to load LightGlue model: {e}", exc_info=True)
                raise
        else:
            logger.debug(f"LightGlue model already loaded, reusing cached instance")

        self._register_model_usage(name)
        return self.models[name]

    def load_depth_anything(self):
        name = 'depth_anything'
        if name not in self.models:
            logger.info(f"Loading Depth-Anything model...")
            self._ensure_vram_available(3000.0)

            try:
                # Try importing Depth-Anything v2 (newer version)
                try:
                    from depth_anything_v2.dpt import DepthAnythingV2

                    model_configs = {
                        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                    }

                    # Use small model by default
                    encoder = self.config.get('depth_anything', {}).get('encoder', 'vits')
                    model = DepthAnythingV2(**model_configs[encoder])
                    model.load_state_dict(
                        torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))

                    logger.info(f"Using Depth-Anything V2 with {encoder} encoder")

                except ImportError:
                    # Fallback to v1
                    from transformers import pipeline

                    model = pipeline(
                        task="depth-estimation",
                        model="LiheYoung/depth-anything-large-hf",
                        device=0 if self.device == 'cuda' else -1
                    )

                    logger.info("Using Depth-Anything V1 from HuggingFace")

                model = model.eval().to(self.device) if hasattr(model, 'eval') else model
                self.models[name] = model

                logger.success(f"Depth-Anything model loaded successfully on {self.device}")

            except Exception as e:
                logger.error(f"Failed to load Depth-Anything model: {e}", exc_info=True)
                raise
        else:
            logger.debug(f"Depth-Anything model already loaded, reusing cached instance")

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
                logger.debug(f"VRAM cache cleared after unloading {model_name}")
        else:
            logger.warning(f"Attempted to unload non-existent model: {model_name}")

    @contextmanager
    def inference_context(self):
        logger.debug("Entering inference context")
        try:
            with torch.no_grad():
                yield
        finally:
            if self.device != 'cpu':
                torch.cuda.empty_cache()
                logger.debug("Exiting inference context, VRAM cache cleared")
