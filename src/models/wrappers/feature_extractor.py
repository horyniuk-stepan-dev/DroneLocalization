import torch
import numpy as np
import cv2
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """Combined feature extraction (SuperPoint + NetVLAD)"""

    def __init__(self, superpoint_model, netvlad_model, device='cuda'):
        self.superpoint = superpoint_model
        self.netvlad = netvlad_model
        self.device = device
        logger.info(f"FeatureExtractor initialized on device: {device}")

    @torch.no_grad()
    def extract_features(self, image: np.ndarray, static_mask: np.ndarray = None) -> dict:
        """
        Extract both local and global features

        Args:
            image: RGB numpy array (H, W, 3)
            static_mask: Binary numpy array (H, W) where 255 is static and 0 is dynamic

        Returns:
            dict with keys:
                - keypoints: (N, 2) array
                - descriptors: (N, 256) array
                - global_desc: (D,) array
        """
        logger.debug(f"Extracting features from image: {image.shape}")

        # Prepare image for SuperPoint (needs grayscale, normalized to [0, 1])
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_tensor = torch.from_numpy(gray_image).float() / 255.0
        gray_tensor = gray_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        # Prepare image for NetVLAD (needs RGB, normalized)
        rgb_tensor = torch.from_numpy(image).float() / 255.0
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 1. Extract local features with SuperPoint
        logger.debug("Extracting local features with SuperPoint...")
        sp_input = {'image': gray_tensor}
        sp_out = self.superpoint(sp_input)

        # SuperPoint returns dict with 'keypoints', 'scores', 'descriptors'
        keypoints = sp_out['keypoints'][0].cpu().numpy()  # (N, 2)
        descriptors = sp_out['descriptors'][0].cpu().numpy()  # (D, N)

        # Transpose descriptors to (N, D) format
        if descriptors.ndim == 2 and descriptors.shape[0] == 256:
            descriptors = descriptors.T

        logger.debug(f"SuperPoint detected {len(keypoints)} keypoints")

        # 2. Filter points by moving object mask
        if static_mask is not None and len(keypoints) > 0:
            logger.debug("Filtering keypoints by static mask...")
            valid_indices = []
            for i, (x, y) in enumerate(keypoints):
                ix, iy = int(round(x)), int(round(y))
                if 0 <= iy < static_mask.shape[0] and 0 <= ix < static_mask.shape[1]:
                    if static_mask[iy, ix] > 128:  # Point on static object
                        valid_indices.append(i)

            if len(valid_indices) > 0:
                keypoints = keypoints[valid_indices]
                descriptors = descriptors[valid_indices]
                logger.debug(f"After filtering: {len(keypoints)} keypoints remain")
            else:
                logger.warning("All keypoints filtered out by mask! Using unfiltered keypoints.")

        # 3. Extract global descriptor with NetVLAD
        logger.debug("Extracting global descriptor with NetVLAD...")
        nv_input = {'image': rgb_tensor}
        nv_out = self.netvlad(nv_input)

        # NetVLAD returns dict with 'global_descriptor'
        if isinstance(nv_out, dict):
            global_desc = nv_out['global_descriptor'][0].cpu().numpy()
        else:
            # If it returns tensor directly
            global_desc = nv_out[0].cpu().numpy()

        logger.debug(f"NetVLAD descriptor shape: {global_desc.shape}")

        logger.success(f"Feature extraction complete: {len(keypoints)} keypoints, global desc dim {len(global_desc)}")

        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'global_desc': global_desc
        }
