import torch
import numpy as np
import cv2
import torchvision.transforms as T
from src.utils.logging_utils import get_logger
from src.utils.image_preprocessor import ImagePreprocessor

logger = get_logger(__name__)


class FeatureExtractor:
    """Combined feature extraction (XFeat + DINOv2)"""

    def __init__(self, local_model, global_model, device='cuda', config=None):
        self.local_model = local_model  # Тепер тут очікується XFeat
        self.global_model = global_model  # DINOv2
        self.device = device
        self.config = config or {}
        self.preprocessor = ImagePreprocessor(config)

        # Трансформації для DINOv2 (ImageNet стандарти)
        # 336×336 замість 224×224 — кратне 14 (patch size DINOv2), дає ~15% краще retrieval
        dino_size = self.config.get('dinov2', {}).get('input_size', 336)
        self.dino_size = dino_size
        self.dinov2_transform = T.Compose([
            T.Resize((dino_size, dino_size), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # FP16 mixed precision для прискорення GPU інференсу (~1.5-2x на GTX 1650)
        self.use_fp16 = (device == 'cuda' and torch.cuda.is_available()
                         and self.config.get('performance', {}).get('fp16', True))
        if self.use_fp16:
            logger.info("FP16 mixed precision ENABLED for inference")

        logger.info(f"FeatureExtractor initialized with XFeat and DINOv2 on {device}")

    @torch.no_grad()
    def extract_global_descriptor(self, image: np.ndarray) -> np.ndarray:
        logger.debug("Extracting global descriptor with DINOv2...")
        dino_tensor = torch.from_numpy(image).float().div_(255.0)
        dino_tensor = dino_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
        dino_input = self.dinov2_transform(dino_tensor)

        # DINOv2 повертає тензор форми (1, 384) — FP16 для прискорення
        if self.use_fp16:
            with torch.cuda.amp.autocast():
                global_desc = self.global_model(dino_input)[0].float().cpu().numpy()
        else:
            global_desc = self.global_model(dino_input)[0].cpu().numpy()
            
        return global_desc

    @torch.no_grad()
    def extract_local_features(self, image: np.ndarray, static_mask: np.ndarray = None) -> dict:
        logger.debug(f"Extracting local features (XFeat) from image: {image.shape}")

        enhanced_image = self.preprocessor.preprocess(image)

        # Підготовка зображення для XFeat
        rgb_tensor = torch.from_numpy(enhanced_image).float().div_(255.0)
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)

        # 1. Витягування локальних ознак через XFeat
        base_top_k = self.config.get('localization', {}).get('xfeat_top_k', 2048)
        img_area = image.shape[0] * image.shape[1]
        adaptive_top_k = max(1024, min(3000, int(base_top_k * (img_area / (640 * 480)))))

        # FP16 для XFeat (якщо увімкнено)
        if self.use_fp16:
            with torch.cuda.amp.autocast():
                xfeat_out = self.local_model.detectAndCompute(rgb_tensor, top_k=adaptive_top_k)[0]
        else:
            xfeat_out = self.local_model.detectAndCompute(rgb_tensor, top_k=adaptive_top_k)[0]

        keypoints = xfeat_out['keypoints'].cpu().numpy()
        descriptors = xfeat_out['descriptors'].cpu().numpy()

        # Bugfix (XFeat 0 points on portrait images): якщо XFeat повернув 0 точок для H > W,
        # спробуємо отримати ознаки з паддінгом до квадрата.
        if len(keypoints) == 0 and image.shape[0] > image.shape[1]:
            logger.warning("XFeat returned 0 points for a portrait image, retrying with square padding...")
            pad_w = image.shape[0] - image.shape[1]
            padded_img = cv2.copyMakeBorder(enhanced_image, 0, 0, 0, pad_w, cv2.BORDER_REFLECT)
            rgb_tensor_padded = torch.from_numpy(padded_img).float().div_(255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.cuda.amp.autocast() if self.use_fp16 else torch.enable_grad(): # Use context conditional properly
                pass
            if self.use_fp16:
                with torch.cuda.amp.autocast():
                    xfeat_out_pad = self.local_model.detectAndCompute(rgb_tensor_padded, top_k=adaptive_top_k)[0]
            else:
                xfeat_out_pad = self.local_model.detectAndCompute(rgb_tensor_padded, top_k=adaptive_top_k)[0]
            
            # Фільтруємо точки, які потрапили на паддінг
            kpts_pad = xfeat_out_pad['keypoints'].cpu().numpy()
            desc_pad = xfeat_out_pad['descriptors'].cpu().numpy()
            if len(kpts_pad) > 0:
                valid_mask = kpts_pad[:, 0] < image.shape[1]
                keypoints = kpts_pad[valid_mask]
                descriptors = desc_pad[valid_mask]

        # 2. Фільтрація точок за маскою динамічних об'єктів (YOLO)
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
        
        logger.success(f"Extracted {len(local_feats['keypoints'])} XFeat keypoints, global DINOv2 desc dim {len(global_desc)}")
        return local_feats