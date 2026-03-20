import torch
import gc
import time
from contextlib import contextmanager
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ModelManager:
    def __init__(self, config=None, device='cuda'):
        self.config = config or {}
        
        use_cuda = self.config.get('models', {}).get('use_cuda', True)
        if not use_cuda:
            logger.info("CUDA force disabled in configuration")
            
        self.device = device if (use_cuda and torch.cuda.is_available() and device == 'cuda') else 'cpu'
        self.models = {}
        self.model_usage = {}
        
        # Конфігурація VRAM
        m_cfg = self.config.get('models', {})
        vram_cfg = m_cfg.get('vram_management', {})
        self.max_vram_ratio = vram_cfg.get('max_vram_ratio', 0.8)
        self.default_vram_required = vram_cfg.get('default_required_mb', 2000.0)

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

    def _ensure_vram_available(self, required_mb: float | None = None):
        if self.device == 'cpu':
            return
        
        req = required_mb if required_mb is not None else self.default_vram_required
        
        while self.get_available_vram_mb() < req and self.models:
            least_used = min(self.model_usage.items(), key=lambda x: x[1])[0]
            logger.warning(f"VRAM insufficient. Unloading least used model: {least_used}")
            self.unload_model(least_used)

    def _register_model_usage(self, name: str):
        self.model_usage[name] = time.time()

    def load_yolo(self):
        name = 'yolo'
        if name not in self.models:
            cfg = self.config.get('models', {}).get(name, {})
            model_path = cfg.get('model_path', 'yolo11x-seg.pt')
            vram_req = cfg.get('vram_required_mb', 1200.0)

            logger.info(f"Loading YOLO model: {model_path}...")
            self._ensure_vram_available(vram_req)
            try:
                from ultralytics import YOLO
                model = YOLO(model_path)
                model.to(self.device)
                self.models[name] = model
                logger.success(f"YOLO model {model_path} loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_xfeat(self):
        name = 'xfeat'
        if name not in self.models:
            cfg = self.config.get('models', {}).get(name, {})
            repo = cfg.get('hub_repo', 'verlab/accelerated_features')
            model_name = cfg.get('hub_model', 'XFeat')
            top_k = cfg.get('top_k', 2048)
            vram_req = cfg.get('vram_required_mb', 300.0)

            logger.info(f"Loading XFeat model ({repo}/{model_name})...")
            self._ensure_vram_available(vram_req)
            try:
                model = torch.hub.load(repo, model_name, pretrained=True, top_k=top_k)
                # FIX: XFeat hardcodes self.dev='cuda' if available, causing crashes if we move to CPU
                if hasattr(model, 'dev'):
                    model.dev = torch.device(self.device)
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
            cfg = self.config.get('models', {}).get(name, {})
            vram_req = cfg.get('vram_required_mb', 500.0)
            
            logger.info(f"Loading SuperPoint model (for LightGlue compatibility)...")
            self._ensure_vram_available(vram_req)
            try:
                from lightglue import SuperPoint
                sp_config = {
                    'nms_radius': cfg.get('nms_radius', 4),
                    'max_num_keypoints': cfg.get('max_keypoints', 4096),
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
            cfg = self.config.get('models', {}).get(name, {})
            vram_req = cfg.get('vram_required_mb', 1000.0)

            logger.info(f"Loading LightGlue model...")
            self._ensure_vram_available(vram_req)
            try:
                from lightglue import LightGlue
                lg_config = {
                    'depth_confidence': cfg.get('depth_confidence', -1),
                    'width_confidence': cfg.get('width_confidence', -1),
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
            cfg = self.config.get('models', {}).get(name, {})
            repo = cfg.get('hub_repo', 'facebookresearch/dinov2')
            model_name = cfg.get('hub_model', 'dinov2_vitl14')
            vram_req = cfg.get('vram_required_mb', 1600.0)

            logger.info(f"Loading DINOv2 ({model_name}) model...")
            self._ensure_vram_available(vram_req)
            try:
                model = torch.hub.load(repo, model_name)
                model = model.eval().to(self.device)
                self.models[name] = model
                logger.success(f"DINOv2 model {model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load DINOv2: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_aliked(self):
        """Завантажує ALIKED extractor (128-dim, lightglue-compatible)"""
        name = 'aliked'
        if name not in self.models:
            cfg = self.config.get('models', {}).get(name, {})
            vram_req = cfg.get('vram_required_mb', 400.0)
            max_keypoints = cfg.get('max_keypoints', 4096)

            logger.info(f"Loading ALIKED model (max_keypoints={max_keypoints})...")
            self._ensure_vram_available(vram_req)
            try:
                from lightglue import ALIKED
                model = ALIKED(max_num_keypoints=max_keypoints).eval().to(self.device)
                self.models[name] = model
                logger.success(f"ALIKED loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load ALIKED: {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_lightglue_aliked(self):
        """Завантажує LightGlue з вагами для ALIKED (128-dim)"""
        name = 'lightglue_aliked'
        if name not in self.models:
            cfg = self.config.get('models', {}).get('lightglue', {})
            vram_req = cfg.get('vram_required_mb', 1000.0)

            logger.info("Loading LightGlue (ALIKED weights)...")
            self._ensure_vram_available(vram_req)
            try:
                from lightglue import LightGlue
                model = LightGlue(
                    features='aliked',
                    depth_confidence=cfg.get('depth_confidence', -1),
                    width_confidence=cfg.get('width_confidence', -1),
                ).eval().to(self.device)
                self.models[name] = model
                logger.success("LightGlue (ALIKED) loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load LightGlue (ALIKED): {e}", exc_info=True)
                raise
        self._register_model_usage(name)
        return self.models[name]

    def load_cesp(self):
        """Завантажує CESP модуль для покращення DINOv2 global descriptors"""
        name = 'cesp'
        if name not in self.models:
            logger.info("Loading CESP module...")
            try:
                from src.models.wrappers.cesp_module import CESP
                cesp_cfg = self.config.get('models', {}).get('cesp', {})
                scales = cesp_cfg.get('scales', [1, 2, 4])
                cesp = CESP(dim=1024, scales=tuple(scales))

                # Завантаження pretrained ваг (якщо є)
                weights_path = cesp_cfg.get('weights_path', None)
                if weights_path:
                    cesp.load_state_dict(torch.load(weights_path, map_location=self.device))
                    logger.success(f"CESP pretrained weights loaded from {weights_path}")
                else:
                    logger.warning("CESP initialized WITHOUT pretrained weights (random init)")

                cesp = cesp.eval().to(self.device)
                self.models[name] = cesp
                logger.success("CESP module loaded")
            except Exception as e:
                logger.error(f"Failed to load CESP: {e}", exc_info=True)
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
