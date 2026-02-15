"""
Feature extraction wrapper (SuperPoint + NetVLAD)
"""

import torch
import numpy as np


class FeatureExtractor:
    """Combined feature extraction"""
    
    def __init__(self, superpoint_model, netvlad_model, device='cuda'):
        self.superpoint = superpoint_model
        self.netvlad = netvlad_model
        self.device = device
    
    @torch.no_grad()
    def extract_features(self, image: np.ndarray, mask: np.ndarray = None) -> dict:
        """
        Extract both local and global features
        
        Returns:
            dict with keys:
                - keypoints: (N, 2) array
                - descriptors: (N, 256) array
                - global_desc: (32768,) array
        """
        # TODO: Preprocess image
        # TODO: Apply mask if provided
        # TODO: Extract SuperPoint features
        # TODO: Extract NetVLAD descriptor
        # TODO: Return combined features
        pass
