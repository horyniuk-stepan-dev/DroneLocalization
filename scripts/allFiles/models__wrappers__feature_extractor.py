import cv2
import numpy as np
import torch
import torchvision.transforms as T
from src.utils.logging_utils import get_logger
from src.utils.image_preprocessor import ImagePreprocessor

logger = get_logger(__name__)


class FeatureExtractor:
    """Combined local (XFeat) + global (DINOv2) feature extraction."""

    def __init__(self, superpoint_model, global_model, device: str = 'cuda', config=None):
        self.superpoint = superpoint_model
        self.global_model = global_model
        self.device = device
        self.preprocessor = ImagePreprocessor(config)

        # Трансформації для DINOv2 (чудово працюють прямо на GPU)
        self.dinov2_transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info(f"FeatureExtractor initialized on device: {device}")

    @torch.no_grad()
    def extract_features(
            self,
            image: np.ndarray,
            static_mask: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        logger.debug(f"Extracting features from image: {image.shape}")

        enhanced = self.preprocessor.preprocess(image)

        # Нормалізація до [0, 1]
        img_f = enhanced.astype(np.float32)
        if img_f.max() > 1.0:
            img_f /= 255.0

        # 1. Локальні ознаки через XFeat (RGB)
        rgb_tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(self.device)
        sp_out = self.superpoint({'image': rgb_tensor})

        keypoints = sp_out['keypoints'][0].cpu().numpy()  # (N, 2)
        descriptors = sp_out['descriptors'][0].cpu().numpy()  # (N, D)

        # ВИДАЛЕНО небезпечну логіку транспонування, XFeatAdapter вже гарантує (N, D)

        # 2. Фільтрація точок за статичною маскою
        if static_mask is not None and len(keypoints) > 0:
            coords = np.round(keypoints).astype(int)
            cx = np.clip(coords[:, 0], 0, static_mask.shape[1] - 1)
            cy = np.clip(coords[:, 1], 0, static_mask.shape[0] - 1)
            valid = static_mask[cy, cx] > 128
            if valid.any():
                keypoints = keypoints[valid]
                descriptors = descriptors[valid]
            else:
                logger.warning("All keypoints filtered out by static mask!")

        # 3. Глобальний дескриптор через DINOv2 (трансформація виконується прямо на GPU)
        dino_input = self.dinov2_transform(rgb_tensor)
        global_desc = self.global_model(dino_input)[0].cpu().numpy()  # (D,)

        logger.success(f"Extracted {len(keypoints)} keypoints, global desc dim: {len(global_desc)}")

        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'global_desc': global_desc,
        }

    @torch.no_grad()
    def extract_local_features(
            self,
            image: np.ndarray,
            static_mask: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Швидке витягування ТІЛЬКИ локальних точок XFeat (для перебору поворотів)"""
        enhanced = self.preprocessor.preprocess(image)

        img_f = enhanced.astype(np.float32)
        if img_f.max() > 1.0:
            img_f /= 255.0

        # Тільки локальні ознаки через XFeat
        rgb_tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(self.device)
        sp_out = self.superpoint({'image': rgb_tensor})

        keypoints = sp_out['keypoints'][0].cpu().numpy()  # (N, 2)
        descriptors = sp_out['descriptors'][0].cpu().numpy()  # (N, 64)

        # Фільтрація за маскою
        if static_mask is not None and len(keypoints) > 0:
            coords = np.round(keypoints).astype(int)
            cx = np.clip(coords[:, 0], 0, static_mask.shape[1] - 1)
            cy = np.clip(coords[:, 1], 0, static_mask.shape[0] - 1)
            valid = static_mask[cy, cx] > 128
            if valid.any():
                keypoints = keypoints[valid]
                descriptors = descriptors[valid]

        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            # Не повертаємо global_desc, щоб зекономити час
        }