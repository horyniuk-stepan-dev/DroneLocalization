import torch
import numpy as np
import cv2


class YOLOWrapper:
    """Wrapper for YOLOv11 segmentation (compatible with YOLOv8 API)"""

    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        # Класи COCO: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 6=train, 7=truck
        self.dynamic_classes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 33, 36, 37}

        # FP16 для YOLO — прискорює інференс на ~40%
        # Ultralytics керує FP16 через параметр half=True при виклику
        self.use_half = (device == 'cuda' and torch.cuda.is_available())

    @torch.no_grad()
    def detect_and_mask(self, image: np.ndarray) -> tuple:
        """
        Detect objects and create static mask

        Returns:
            static_mask: Binary mask of static areas (255 for static, 0 for dynamic)
            detections: List of detection dicts
        """
        # verbose=False вимикає зайве логування кожного кадру в консоль
        # half=True для FP16 інференсу
        results = self.model(image, verbose=False, half=self.use_half)
        result = results[0]

        height, width = image.shape[:2]
        static_mask = np.ones((height, width), dtype=np.uint8) * 255

        detections = []

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            # Vectorized: знайти всі динамічні маски одразу
            dynamic_mask_indices = [i for i, cls in enumerate(classes) if cls in self.dynamic_classes]

            for i, (cls, conf, box) in enumerate(zip(classes, confidences, boxes)):
                detections.append({
                    "class_id": int(cls),
                    "confidence": float(conf),
                    "bbox": box[:4].tolist()
                })

            # Об'єднуємо всі динамічні маски за один раз
            if dynamic_mask_indices:
                combined_dynamic = np.zeros((height, width), dtype=np.float32)
                for idx in dynamic_mask_indices:
                    mask_resized = cv2.resize(masks[idx], (width, height), interpolation=cv2.INTER_NEAREST)
                    combined_dynamic = np.maximum(combined_dynamic, mask_resized)
                static_mask[combined_dynamic > 0.5] = 0

        return static_mask, detections