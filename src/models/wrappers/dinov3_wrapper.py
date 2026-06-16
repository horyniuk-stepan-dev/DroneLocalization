"""
DINOv3 Wrapper for DroneLocalization.

Replaces the standard DINOv2 (torch.hub) with facebook/dinov3-vitl16-pretrain-sat493m
loaded via HuggingFace Transformers.

Key differences from DINOv2:
- Loaded via transformers.AutoModel (not torch.hub)
- input_size: 224 (vs 336 for DINOv2)
- patch_size: 16 (vs 14)
- normalization: satellite-domain stats (vs ImageNet)
- hidden_size: 1024 — SAME → existing databases are fully compatible

Usage (drop-in for model_manager.load_dinov2):
    wrapper = DINOv3Wrapper("facebook/dinov3-vitl16-pretrain-sat493m", device="cuda")
    cls_token = wrapper(image_tensor)  # (B, 1024)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_HF_MODEL_ID = "facebook/dinov3-vitl16-pretrain-sat493m"


class DINOv3Wrapper(nn.Module):
    """
    Drop-in wrapper for facebook/dinov3-vitl16-pretrain-sat493m.

    Exposes the same call interface as DINOv2 from torch.hub:
        forward(pixel_values) -> cls_token  # shape (B, 1024)
        forward_features(pixel_values) -> dict with 'x_norm_patchtokens' and 'x_norm_clstoken'

    This allows FeatureExtractor and CespModule to work without any changes.
    """

    def __init__(self, model_id: str = _HF_MODEL_ID, device: str = "cuda"):
        super().__init__()
        from transformers import AutoModel

        logger.info(f"Loading DINOv3 from HuggingFace: {model_id} ...")
        # trust_remote_code needed for custom DINOv3ViTModel architecture
        self._model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self._model = self._model.eval().to(device)
        self._device = device

        hidden_size = self._model.config.hidden_size
        logger.info(
            f"DINOv3 loaded: hidden_size={hidden_size}, "
            f"image_size={self._model.config.image_size}, "
            f"patch_size={self._model.config.patch_size}"
        )

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning the [CLS] token — identical interface to DINOv2.

        Args:
            pixel_values: (B, 3, H, W) float tensor, already normalized.

        Returns:
            cls_token: (B, 1024) float tensor.
        """
        outputs = self._model(pixel_values=pixel_values)
        # HuggingFace ViT models expose last_hidden_state: (B, 1 + num_patches, hidden)
        # Index 0 is the [CLS] token
        cls_token = outputs.last_hidden_state[:, 0, :]
        return cls_token

    @torch.no_grad()
    def forward_features(self, pixel_values: torch.Tensor) -> dict:
        """
        Returns patch tokens and CLS token — compatible with CESP module.

        Returns dict with:
            'x_norm_clstoken':    (B, 1024)
            'x_norm_patchtokens': (B, num_patches, 1024)
        """
        outputs = self._model(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state  # (B, 1 + N_patches, 1024)
        return {
            "x_norm_clstoken": hidden[:, 0, :],
            "x_norm_patchtokens": hidden[:, 1:, :],
        }

    def to(self, *args, **kwargs):
        self._model = self._model.to(*args, **kwargs)
        return self

    def eval(self):
        self._model = self._model.eval()
        return self

    def train(self, mode: bool = True):
        # Keep model frozen — DINOv3 is always used in inference mode
        self._model = self._model.eval()
        return self

    def parameters(self, recurse: bool = True):
        return self._model.parameters(recurse=recurse)
