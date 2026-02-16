import torch
import numpy as np
import cv2

class YOLOWrapper:
    """Wrapper for YOLOv8 segmentation"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.dynamic_classes = {0, 1, 2, 3, 5, 6, 7}
    
    @torch.no_grad()
    def detect_and_mask(self, image: np.ndarray) -> tuple:
        """
        Detect objects and create static mask
        
        Returns:
            static_mask: Binary mask of static areas (255 for static, 0 for dynamic)
            detections: List of detection dicts
        """
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
                
                if class_id in self.dynamic_classes:
                    mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    static_mask[mask_resized > 0.5] = 0
                    
        return static_mask, detections