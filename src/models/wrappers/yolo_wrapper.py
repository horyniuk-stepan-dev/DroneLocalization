"""
YOLOv8-Seg wrapper
"""

import torch
import numpy as np


class YOLOWrapper:
    """Wrapper for YOLOv8 segmentation"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    @torch.no_grad()
    def detect_and_mask(self, image: np.ndarray) -> tuple:
        """
        Detect objects and create static mask
        
        Returns:
            masks: Binary mask of dynamic objects
            detections: List of detection dicts
        """
        # TODO: Run inference
        # TODO: Extract masks
        # TODO: Create combined static mask
        # TODO: Return masks and detections
        pass
