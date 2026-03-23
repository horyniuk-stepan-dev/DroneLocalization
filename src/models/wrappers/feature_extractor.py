import torch
import numpy as np
import cv2
import torchvision.transforms as T
from src.utils.logging_utils import get_logger
from src.utils.image_preprocessor import ImagePreprocessor

logger = get_logger(__name__)


class FeatureExtractor:
    """Combined feature extraction (ALIKED + DINOv2 [+ CESP])"""

    def __init__(self, local_model, global_model, device='cuda', config=None, cesp_module=None):
        self.local_model = local_model  # ALIKED
        self.global_model = global_model  # DINOv2
        self.device = device
        self.config = config or {}
        self.preprocessor = ImagePreprocessor(config)
        self.cesp_module = cesp_module  # Опціональний CESP для покращення global descriptors

        # Трансформації для DINOv2 (ImageNet стандарти)
        dino_size = self.config.get('dinov2', {}).get('input_size', 336)
        self.dino_size = dino_size
        self.dinov2_transform = T.Compose([
            T.Resize((dino_size, dino_size), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # FP16 mixed precision для прискорення GPU інференсу
        self.use_fp16 = (device == 'cuda' and torch.cuda.is_available()
                         and self.config.get('performance', {}).get('fp16', True))
        if self.use_fp16:
            logger.info("FP16 mixed precision ENABLED for inference")

        cesp_status = "with CESP" if cesp_module is not None else "without CESP"
        logger.info(f"FeatureExtractor initialized with ALIKED and DINOv2 ({cesp_status}) on {device}")

    @torch.no_grad()
    def extract_global_descriptor(self, image: np.ndarray) -> np.ndarray:
        logger.debug("Extracting global descriptor with DINOv2...")
        dino_tensor = torch.from_numpy(image).float().div_(255.0)
        dino_tensor = dino_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
        dino_input = self.dinov2_transform(dino_tensor)

        if self.cesp_module is not None:
            # CESP mode: отримуємо patch tokens замість CLS
            if self.use_fp16:
                with torch.cuda.amp.autocast():
                    features = self.global_model.forward_features(dino_input)
                    patch_tokens = features['x_norm_patchtokens'].float()
            else:
                features = self.global_model.forward_features(dino_input)
                patch_tokens = features['x_norm_patchtokens']

            h_patches = self.dino_size // 14
            w_patches = self.dino_size // 14
            global_desc = self.cesp_module(patch_tokens, h_patches, w_patches)[0].cpu().numpy()
        else:
            # Стандартний mode: CLS token
            if self.use_fp16:
                with torch.cuda.amp.autocast():
                    global_desc = self.global_model(dino_input)[0].float().cpu().numpy()
            else:
                global_desc = self.global_model(dino_input)[0].cpu().numpy()

        return global_desc

    @torch.no_grad()
    def extract_local_features(self, image: np.ndarray, static_mask: np.ndarray = None) -> dict:
        logger.debug(f"Extracting local features (ALIKED) from image: {image.shape}")

        enhanced_image = self.preprocessor.preprocess(image)

        # Підготовка зображення для ALIKED (LightGlue format)
        rgb_tensor = torch.from_numpy(enhanced_image).float().div_(255.0)
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)

        # ALIKED очікує словник зі списком/тензором 'image'
        input_dict = {'image': rgb_tensor}

        if self.use_fp16:
            with torch.cuda.amp.autocast():
                aliked_out = self.local_model(input_dict)
        else:
            aliked_out = self.local_model(input_dict)

        # LightGlue ALIKED wrapper повертає батч: (1, N, 2) та (1, N, 128)
        keypoints = aliked_out['keypoints'][0].cpu().numpy()
        descriptors = aliked_out['descriptors'][0].cpu().numpy()

        # Фільтрація точок за маскою динамічних об'єктів (YOLO)
        if static_mask is not None and len(keypoints) > 0:
            # Vectorized YOLO mask filtering
            ix = np.round(keypoints[:, 0]).astype(np.intp)
            iy = np.round(keypoints[:, 1]).astype(np.intp)
            in_bounds = (iy >= 0) & (iy < static_mask.shape[0]) & (ix >= 0) & (ix < static_mask.shape[1])
            valid = np.zeros(len(keypoints), dtype=bool)
            valid[in_bounds] = static_mask[iy[in_bounds], ix[in_bounds]] > 128

            if valid.any():
                keypoints = keypoints[valid]
                descriptors = descriptors[valid]
            else:
                logger.warning("All keypoints filtered out by YOLO mask!")

        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'coords_2d': keypoints.copy()
        }

    @torch.no_grad()
    def extract_features(self, image: np.ndarray, static_mask: np.ndarray = None) -> dict:
        local_feats = self.extract_local_features(image, static_mask)
        global_desc = self.extract_global_descriptor(image)
        local_feats['global_desc'] = global_desc
        
        logger.success(f"Extracted {len(local_feats['keypoints'])} ALIKED keypoints, global DINOv2 desc dim {len(global_desc)}")
        return local_feats