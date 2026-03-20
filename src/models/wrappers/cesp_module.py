import torch
import torch.nn as nn
import torch.nn.functional as F


class CESP(nn.Module):
    """Cross-Enhancement Spatial Pyramid для DINOv2 patch tokens.
    
    IEEE RA-L 2025: "DINOv2-based UAV Visual Self-localization"
    Покращує multi-scale сприйняття для aerial imagery.
    
    Вхід: patch_tokens (B, N, D) з DINOv2
    Вихід: enhanced_descriptor (B, D) — L2-нормалізований
    
    Примітка: потребує навчання на парах UAV↔satellite зображень.
    Без навчених ваг повертає усереднення multi-scale features (random projection).
    """

    def __init__(self, dim: int = 1024, scales: tuple = (1, 2, 4)):
        super().__init__()
        self.dim = dim
        self.scales = scales

        # Проекційні шари для кожного масштабу піраміди
        self.projectors = nn.ModuleList([
            nn.Linear(dim, dim) for _ in scales
        ])

        # Фінальне злиття (N_scales * dim → dim)
        self.fusion = nn.Sequential(
            nn.Linear(len(scales) * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, patch_tokens: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, N, D) — patch tokens з DINOv2 (без CLS)
            h_patches: кількість патчів по висоті (для 336×336 + patch_size=14 → 24)
            w_patches: кількість патчів по ширині

        Returns:
            enhanced: (B, D) — L2-нормалізований глобальний дескриптор
        """
        B, N, D = patch_tokens.shape
        # Reshape до 2D просторової сітки: (B, D, H, W)
        x = patch_tokens.reshape(B, h_patches, w_patches, D).permute(0, 3, 1, 2)

        scale_features = []
        for scale, proj in zip(self.scales, self.projectors):
            if scale == 1:
                # Глобальне усереднення всіх патчів
                pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, D)
            else:
                # Spatial Pyramid: розбити на scale×scale регіонів → усереднити
                pooled = F.adaptive_avg_pool2d(x, scale)  # (B, D, scale, scale)
                pooled = pooled.flatten(2).mean(dim=2)     # (B, D)
            scale_features.append(proj(pooled))

        # Cross-Enhancement: конкатенація + fusion
        multi_scale = torch.cat(scale_features, dim=1)  # (B, N_scales*D)
        enhanced = self.fusion(multi_scale)              # (B, D)
        enhanced = F.normalize(enhanced, p=2, dim=1)     # L2 нормалізація

        return enhanced
