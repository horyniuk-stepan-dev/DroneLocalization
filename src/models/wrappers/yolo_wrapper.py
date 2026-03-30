import cv2
import numpy as np
import torch

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class YOLOWrapper:
    """Wrapper for YOLOv11 segmentation (compatible with YOLOv8 API)"""

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        # Класи COCO: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
        self.dynamic_classes = {0, 1, 2, 3, 5, 7}

        # FP16 для YOLO — прискорює інференс на ~40%
        # Ultralytics керує FP16 через параметр half=True при виклику
        self.use_half = device == "cuda" and torch.cuda.is_available()

    @torch.no_grad()
    def detect_and_mask(self, image: np.ndarray) -> tuple:
        """
        Detect objects and create static mask (single image).
        Делегує до batch-методу для уникнення дублювання логіки.

        Returns:
            static_mask: Binary mask of static areas (255 for static, 0 for dynamic)
            detections: List of detection dicts
        """
        results = self.detect_and_mask_batch([image])
        if not results:
            return np.ones(image.shape[:2], dtype=np.uint8) * 255, []
        return results[0]

    @torch.no_grad()
    def detect_and_mask_batch(self, images: list[np.ndarray]) -> list[tuple]:
        """
        Detect objects and create static masks for a batch of images.
        Повертає list[(static_mask, detections)] того самого порядку.
        """
        if not images:
            return []

        # YOLO expects list of images or a 4D tensor (B, H, W, 3). Ultralytics natively handles list of ndarrays.
        # half=True для FP16 інференсу
        # conf=0.50 відкидає слабкі передбачення
        results = self.model(images, verbose=False, half=self.use_half, conf=0.50)

        output = []
        MAX_SINGLE_MASK_RATIO = 0.40
        MAX_COMBINED_MASK_RATIO = 0.70

        for result, image in zip(results, images):
            height, width = image.shape[:2]
            static_mask = np.ones((height, width), dtype=np.uint8) * 255
            detections = []
            total_pixels = height * width

            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()

                dynamic_mask_indices = [
                    i for i, cls in enumerate(classes) if cls in self.dynamic_classes
                ]

                for i, (cls, conf, box) in enumerate(zip(classes, confidences, boxes)):
                    detections.append(
                        {"class_id": int(cls), "confidence": float(conf), "bbox": box[:4].tolist()}
                    )

                if dynamic_mask_indices:
                    combined_dynamic = np.zeros((height, width), dtype=np.float32)
                    for idx in dynamic_mask_indices:
                        mask_resized = cv2.resize(
                            masks[idx], (width, height), interpolation=cv2.INTER_NEAREST
                        )
                        mask_area = np.sum(mask_resized > 0.5)
                        if mask_area / total_pixels > MAX_SINGLE_MASK_RATIO:
                            continue
                        combined_dynamic = np.maximum(combined_dynamic, mask_resized)

                    combined_area = np.sum(combined_dynamic > 0.5)
                    if combined_area / total_pixels < MAX_COMBINED_MASK_RATIO:
                        static_mask[combined_dynamic > 0.5] = 0
                    else:
                        logger.warning(
                            f"YOLO OVER-MASKING DETECTED ({combined_area / total_pixels:.2%}). "
                            "Frame preserved."
                        )

            output.append((static_mask, detections))

        return output
