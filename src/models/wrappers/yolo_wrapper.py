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

    @torch.no_grad()
    def detect_and_mask(self, image: np.ndarray) -> tuple:
        """
        Detect objects and create static mask

        Returns:
            static_mask: Binary mask of static areas (255 for static, 0 for dynamic)
            detections: List of detection dicts
        """
        # verbose=False вимикає зайве логування кожного кадру в консоль
        results = self.model(image, verbose=False)
        result = results[0]

        height, width = image.shape[:2]
        static_mask = np.ones((height, width), dtype=np.uint8) * 255

        detections = []

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for mask, cls, conf, box in zip(masks, classes, confidences, boxes):
                class_id = int(cls)
                detections.append({
                    "class_id": class_id,
                    "confidence": float(conf),
                    "bbox": box[:4].tolist()
                })

                # Відсікаємо лише рухомі об'єкти (авто, люди тощо)
                if class_id in self.dynamic_classes:
                    mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    static_mask[mask_resized > 0.5] = 0

        return static_mask, detections