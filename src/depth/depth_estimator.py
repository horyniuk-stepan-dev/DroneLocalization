"""Lightweight depth estimation wrapper для scale recovery.

Підтримує:
  - Depth Anything V2 (рекомендовано — краща точність)
  - MiDaS v3 (fallback — менші вимоги до VRAM)
  - Dummy estimator (для тестів без GPU)

Використання в pipeline:
  depth_est = DepthEstimator.build("depth_anything_v2", device="cuda")
  scale = depth_est.get_relative_scale(frame_rgb)  # float
"""


import cv2
import numpy as np
import torch

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DepthEstimator:
    """Абстрактний depth estimator. Використовуй DepthEstimator.build()."""

    def estimate(self, image_rgb: np.ndarray) -> np.ndarray:
        """Повертає depth map (H, W) float32, відносні значення."""
        raise NotImplementedError

    def get_relative_scale(self, image_rgb: np.ndarray) -> float:
        """Повертає скалярний відносний масштаб (1/median_depth).

        Менше значення = об'єкт далі (більша висота) = менший GSD.
        Більше значення = об'єкт ближче (менша висота) = більший GSD.
        """
        depth = self.estimate(image_rgb)
        # Беремо центральний регіон (ігноруємо краї де artifacts)
        h, w = depth.shape
        cx1, cx2 = w // 4, 3 * w // 4
        cy1, cy2 = h // 4, 3 * h // 4
        center_depth = depth[cy1:cy2, cx1:cx2]

        # Маска валідних значень (не нулі)
        valid_mask = center_depth > 0
        if not np.any(valid_mask):
            return 1.0

        median_d = float(np.median(center_depth[valid_mask]))
        if median_d < 1e-6:
            return 1.0

        return 1.0 / median_d  # відносний scale: далі = менше

    @staticmethod
    def build(backend: str = "depth_anything_v2", device: str = "cuda") -> "DepthEstimator":
        if backend == "depth_anything_v2":
            return _DepthAnythingV2Estimator(device)
        elif backend == "midas":
            return _MiDaSEstimator(device)
        elif backend == "dummy":
            return _DummyDepthEstimator()
        else:
            raise ValueError(f"Unknown depth backend: {backend}")


class _DepthAnythingV2Estimator(DepthEstimator):
    """Depth Anything V2 Small/Base/Large wrapper."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            try:
                from depth_anything_v2.dpt import DepthAnythingV2
            except ImportError:
                # Шукаємо у third_party/Depth-Anything-V2
                import os
                import sys
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                local_path = os.path.join(project_root, "third_party", "Depth-Anything-V2")
                if os.path.exists(local_path):
                    sys.path.append(local_path)
                    from depth_anything_v2.dpt import DepthAnythingV2
                else:
                    raise ImportError(f"Depth-Anything-V2 not found in {local_path}") from None

            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }

            # Визначаємо тип енкодера (за замовчуванням vits для швидкості)
            encoder = 'vits'
            self._model = DepthAnythingV2(**model_configs[encoder])

            import os
            # Шукаємо ваги за різними можливими іменами
            weight_names = [
                f"depth_anything_v2_{encoder}.pth",
                "depth_anything_v2_vits.pth"
            ]

            from config.paths import models_root

            # Single models root: <repo>/models in dev (cwd-independent),
            # <_MEIPASS>/models in a frozen build. Bare "models" stays as a
            # cwd-relative fallback for unusual invocations.
            model_dirs = [str(models_root()), "models"]
            weight_paths = []
            for name in weight_names:
                for d in model_dirs:
                    weight_paths.append(os.path.join(d, name))
                weight_paths.append(os.path.expanduser(f"~/.cache/depth_anything_v2/{name}"))

            for wp in weight_paths:
                if os.path.exists(wp):
                    self._model.load_state_dict(
                        torch.load(wp, map_location='cpu', weights_only=True)
                    )
                    logger.info(f"Depth Anything V2 weights loaded from {wp}")
                    break
            else:
                logger.warning(
                    f"Depth Anything V2 weights not found. Searched in: {weight_paths}"
                )

            self._model = self._model.to(self.device).eval()
            logger.info(f"Depth Anything V2 ({encoder}) initialized on {self.device}")
        except ImportError:
            logger.error("depth_anything_v2 not installed. Fallback to MiDaS is recommended.")
            raise

    @torch.no_grad()
    def estimate(self, image_rgb: np.ndarray) -> np.ndarray:
        try:
            self._lazy_load()
            # Depth Anything очікує BGR для infer_image
            bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            depth = self._model.infer_image(bgr)
            return depth.astype(np.float32)
        except Exception as e:
            logger.error(f"Depth Anything V2 inference failed: {e}")
            return np.ones(image_rgb.shape[:2], dtype=np.float32)


class _MiDaSEstimator(DepthEstimator):
    """MiDaS v3 через torch.hub."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._transform = None

    def _lazy_load(self):
        if self._model is not None:
            return
        logger.info("Loading MiDaS DPT_BEiT_L_512 from torch.hub...")
        self._model = torch.hub.load("intel-isl/MiDaS", "DPT_BEiT_L_512")
        self._model.to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self._transform = transforms.beit512_transform
        logger.info("MiDaS loaded successfully")

    @torch.no_grad()
    def estimate(self, image_rgb: np.ndarray) -> np.ndarray:
        try:
            self._lazy_load()
            # MiDaS очікує BGR
            bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            input_batch = self._transform(bgr).to(self.device)
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            return prediction.cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.error(f"MiDaS inference failed: {e}")
            return np.ones(image_rgb.shape[:2], dtype=np.float32)


class _DummyDepthEstimator(DepthEstimator):
    """Заглушка для систем без GPU."""
    def estimate(self, image_rgb: np.ndarray) -> np.ndarray:
        return np.ones(image_rgb.shape[:2], dtype=np.float32)
