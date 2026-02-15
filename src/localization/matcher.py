"""
Feature matching using LightGlue
"""

import torch
import numpy as np


class FeatureMatcher:
    """Matches features between query and reference"""
    
    def __init__(self, lightglue_model, device='cuda'):
        self.lightglue = lightglue_model
        self.device = device
    
    @torch.no_grad()
    def match(self, query_features, ref_features):
        """Match features between frames"""
        # TODO: Prepare tensors
        # TODO: Run LightGlue
        # TODO: Extract matches
        # TODO: Return matched keypoint pairs
        pass
