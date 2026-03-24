import numpy as np
import torch

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ALIKEDWrapper:
    """ALIKED feature extractor для LightGlue fallback.

    ALIKED видає 128-dim дескриптори (vs SuperPoint 256-dim).
    LightGlue має офіційні pretrained ваги для ALIKED.
    """

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def extract(self, image_tensor: torch.Tensor) -> dict:
        """Екстракція ALIKED features з тензору зображення (lightglue format)."""
        return self.model.extract(image_tensor)

    @torch.no_grad()
    def extract_from_numpy(self, image_rgb: np.ndarray, static_mask: np.ndarray = None) -> dict:
        """Екстракція з numpy RGB зображення + фільтрація за YOLO маскою.

        Returns:
            dict з ключами: keypoints (1, K, 2), descriptors (1, K, 128)
        """
        from lightglue.utils import numpy_image_to_torch

        tensor = numpy_image_to_torch(image_rgb).to(self.device)
        features = self.model.extract(tensor)

        # Фільтрація за маскою динамічних об'єктів
        if static_mask is not None and "keypoints" in features:
            kpts = features["keypoints"][0].cpu().numpy()
            if len(kpts) > 0:
                ix = np.round(kpts[:, 0]).astype(np.intp)
                iy = np.round(kpts[:, 1]).astype(np.intp)
                h, w = static_mask.shape[:2]
                in_bounds = (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
                valid = np.zeros(len(kpts), dtype=bool)
                valid[in_bounds] = static_mask[iy[in_bounds], ix[in_bounds]] > 128

                if valid.any():
                    valid_t = torch.from_numpy(valid).to(self.device)
                    filtered = {
                        "keypoints": features["keypoints"][:, valid_t],
                        "descriptors": features["descriptors"][:, valid_t],
                    }
                    # Зберігаємо keypoint_scores якщо є
                    if "keypoint_scores" in features and features["keypoint_scores"] is not None:
                        filtered["keypoint_scores"] = features["keypoint_scores"][:, valid_t]
                    features = filtered
                    logger.debug(
                        f"ALIKED: {int(valid.sum())}/{len(kpts)} keypoints after mask filter"
                    )

        return features
