import torch
import numpy as np
import cv2
import torchvision.transforms as T
from src.utils.logging_utils import get_logger
from src.utils.image_preprocessor import ImagePreprocessor

logger = get_logger(__name__)


class FeatureExtractor:
    """Combined feature extraction (SuperPoint + DINOv2)"""

    def __init__(self, superpoint_model, global_model, device='cuda', config=None):
        self.superpoint = superpoint_model
        self.global_model = global_model
        self.device = device
        self.preprocessor = ImagePreprocessor(config)

        # Трансформації для DINOv2 (ImageNet стандарти та розмір 224x224)
        self.dinov2_transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        logger.info(f"FeatureExtractor initialized on device: {device}")

    @torch.no_grad()
    def extract_features(self, image: np.ndarray, static_mask: np.ndarray = None) -> dict:
        logger.debug(f"Extracting features from image: {image.shape}")

        enhanced_image = self.preprocessor.preprocess(image)

        # 1. Підготовка для SuperPoint (ЧБ)
        gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2GRAY)
        gray_tensor = torch.from_numpy(gray_image).float() / 255.0
        gray_tensor = gray_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        # 2. Підготовка для DINOv2 (Колір, RGB тензор)
        rgb_tensor = torch.from_numpy(enhanced_image).float() / 255.0
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 3. Витягування локальних ознак SuperPoint
        sp_input = {'image': gray_tensor}
        sp_out = self.superpoint(sp_input)

        keypoints = sp_out['keypoints'][0].cpu().numpy()
        descriptors = sp_out['descriptors'][0].cpu().numpy()

        if descriptors.ndim == 2 and descriptors.shape[0] == 256:
            descriptors = descriptors.T

        # 4. Фільтрація точок за маскою (якщо є)
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
                logger.warning("All keypoints filtered out by mask!")

        # 5. Витягування ГЛОБАЛЬНОГО дескриптора з DINOv2
        logger.debug("Extracting global descriptor with DINOv2...")
        dino_input = self.dinov2_transform(rgb_tensor)

        # DINOv2 повертає тензор форми (1, 384). Беремо нульовий індекс.
        global_desc = self.global_model(dino_input)[0].cpu().numpy()

        logger.success(f"Extracted {len(keypoints)} keypoints, global desc dim {len(global_desc)}")

        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'global_desc': global_desc,
            'coords_2d': keypoints.copy()
        }