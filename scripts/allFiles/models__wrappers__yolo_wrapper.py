import numpy as np
import cv2
import torch
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# COCO class IDs for dynamic objects:
# 0: person, 1: bicycle, 2: car, 3: motorcycle,
# 5: bus, 6: train, 7: truck
DYNAMIC_COCO_CLASSES: frozenset[int] = frozenset({0, 1, 2, 3, 5, 6, 7})


class YOLOWrapper:
    """Wrapper for YOLO11 segmentation — filters dynamic objects for static mask generation."""

    def __init__(self, model, device: str = 'cuda', confidence_threshold: float = 0.4):
        self.model = model
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.dynamic_classes = DYNAMIC_COCO_CLASSES

    def detect_and_mask(self, image: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Detect dynamic objects and produce a binary static-area mask.

        Returns:
            static_mask: uint8 array (255 = static, 0 = dynamic object)
            detections:  list of dicts with class_id, confidence, bbox, is_dynamic
        """
        results = self.model(image, verbose=False, device=self.device)
        result = results[0]

        height, width = image.shape[:2]
        static_mask = np.full((height, width), 255, dtype=np.uint8)
        detections = []

        if result.masks is None:
            return static_mask, detections

        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.data.cpu().numpy()

        for mask, cls, conf, box in zip(masks, classes, confidences, boxes):
            class_id = int(cls)
            is_dynamic = class_id in self.dynamic_classes

            detections.append({
                'class_id': class_id,
                'confidence': float(conf),
                'bbox': box[:4].tolist(),
                'is_dynamic': is_dynamic,
            })

            if is_dynamic and conf >= self.confidence_threshold:
                mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                static_mask[mask_resized > 0.5] = 0

        return static_mask, detections
