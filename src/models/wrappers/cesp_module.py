import torch
import torch.nn as nn
import torch.nn.functional as F


class CESP(nn.Module):
    """Cross-Enhancement Spatial Pyramid for DINOv2 patch tokens.

    IEEE RA-L 2025: "DINOv2-based UAV Visual Self-localization"
    Enhances multi-scale perception for aerial imagery.

    Input: patch_tokens (B, N, D) from DINOv2
    Output: enhanced_descriptor (B, D) — L2-normalized
    """

    def __init__(self, dim: int = 1024, scales: tuple = (1, 2, 4)):
        super().__init__()
        self.dim = dim
        self.scales = scales

        # Projection layers for each pyramid scale
        self.projectors = nn.ModuleList([nn.Linear(dim, dim) for _ in scales])

        # Final fusion layer (N_scales * dim -> dim)
        self.fusion = nn.Sequential(
            nn.Linear(len(scales) * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, patch_tokens: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, N, D) — patch tokens from DINOv2 (excluding CLS)
            h_patches: number of patches along height
            w_patches: number of patches along width

        Returns:
            enhanced: (B, D) — L2-normalized global descriptor
        """
        B, N, D = patch_tokens.shape
        # Reshape to 2D spatial grid: (B, D, H, W)
        x = patch_tokens.reshape(B, h_patches, w_patches, D).permute(0, 3, 1, 2)

        scale_features = []
        for scale, proj in zip(self.scales, self.projectors):
            if scale == 1:
                # Global average pooling of all patches
                pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, D)
            else:
                # Spatial Pyramid: pool into scale x scale regions
                pooled = F.adaptive_avg_pool2d(x, scale)  # (B, D, scale, scale)
                pooled = pooled.flatten(2).mean(dim=2)  # (B, D)
            scale_features.append(proj(pooled))

        # Cross-Enhancement: concatenation + fusion
        multi_scale = torch.cat(scale_features, dim=1)  # (B, N_scales*D)
        enhanced = self.fusion(multi_scale)  # (B, D)
        enhanced = F.normalize(enhanced, p=2, dim=1)  # L2 normalization

        return enhanced
