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
        self.preprocessor = ImagePreprocessor(config)

        # Трансформації для DINOv2 (ImageNet стандарти та розмір 224x224)
        self.dinov2_transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        logger.info(f"FeatureExtractor initialized with XFeat and DINOv2 on {device}")

    @torch.no_grad()
    def extract_features(self, image: np.ndarray, static_mask: np.ndarray = None) -> dict:
        logger.debug(f"Extracting features from image: {image.shape}")

        enhanced_image = self.preprocessor.preprocess(image)

        # Підготовка зображення для XFeat та DINOv2 (Обидвом потрібен RGB тензор)
        rgb_tensor = torch.from_numpy(enhanced_image).float() / 255.0
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 1. Витягування локальних ознак через XFeat
        # top_k можна винести в конфіг, тут залишаємо 2048 для порівняння з SuperPoint
        xfeat_out = self.local_model.detectAndCompute(rgb_tensor, top_k=2048)[0]

        keypoints = xfeat_out['keypoints'].cpu().numpy()
        descriptors = xfeat_out['descriptors'].cpu().numpy()

        # 2. Фільтрація точок за маскою динамічних об'єктів (YOLO)
        if static_mask is not None and len(keypoints) > 0:
            valid_indices = []
            for i, (x, y) in enumerate(keypoints):
                ix, iy = int(round(x)), int(round(y))
                if 0 <= iy < static_mask.shape[0] and 0 <= ix < static_mask.shape[1]:
                    if static_mask[iy, ix] > 128:
                        valid_indices.append(i)

            if len(valid_indices) > 0:
                keypoints = keypoints[valid_indices]
                descriptors = descriptors[valid_indices]
            else:
                logger.warning("All keypoints filtered out by YOLO mask!")

        # 3. Витягування ГЛОБАЛЬНОГО дескриптора з DINOv2
        # DINOv2 — семантична модель, їй потрібне натуральне зображення без CLAHE
        logger.debug("Extracting global descriptor with DINOv2...")
        dino_tensor = torch.from_numpy(image).float() / 255.0
        dino_tensor = dino_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        dino_input = self.dinov2_transform(dino_tensor)

        # DINOv2 повертає тензор форми (1, 384)
        global_desc = self.global_model(dino_input)[0].cpu().numpy()

        logger.success(f"Extracted {len(keypoints)} XFeat keypoints, global DINOv2 desc dim {len(global_desc)}")

        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'global_desc': global_desc,
            'coords_2d': keypoints.copy()
        }