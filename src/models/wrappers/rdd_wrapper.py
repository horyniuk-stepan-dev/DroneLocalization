import sys
from pathlib import Path

import torch

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Lazy import RDD — потребує git clone https://github.com/xtcpete/rdd
_RDD_BUILD = None


def _import_rdd():
    """Lazy import RDD з third-party або models/rdd."""
    global _RDD_BUILD
    if _RDD_BUILD is not None:
        return _RDD_BUILD

    # Пошук RDD пакету у кількох місцях
    search_paths = [
        Path(__file__).resolve().parents[3] / "third_party" / "rdd",  # <project>/third_party/rdd
        Path(__file__).resolve().parents[3] / "models" / "rdd",       # <project>/models/rdd
    ]

    for rdd_path in search_paths:
        if rdd_path.exists() and (rdd_path / "RDD" / "RDD.py").exists():
            if str(rdd_path) not in sys.path:
                sys.path.insert(0, str(rdd_path))
            break

    try:
        from RDD.RDD import build as rdd_build
        from RDD.utils import read_config
        _RDD_BUILD = (rdd_build, read_config, rdd_path)
        logger.info("RDD module imported successfully")
        return _RDD_BUILD
    except ImportError as e:
        logger.error(
            f"RDD import failed: {e}. "
            f"Install: git clone --recursive https://github.com/xtcpete/rdd third_party/rdd"
        )
        raise


class RDDWrapper:
    """RDD (Robust Deformable Detector) wrapper — drop-in замість ALIKED.

    RDD використовує deformable transformers для scale-invariant
    детекції keypoints та побудови дескрипторів.

    Output format ідентичний ALIKED:
        {"keypoints": (1, N, 2), "descriptors": (1, N, D)}

    Usage:
        wrapper = RDDWrapper(weights_path="models/rdd.pth", device="cuda")
        model = wrapper.model  # Pass to FeatureExtractor as local_model
    """

    def __init__(self, weights_path: str = None, device: str = "cuda", max_keypoints: int = 4096):
        self.device = device
        self.max_keypoints = max_keypoints

        build_fn, read_config_fn, rdd_path = _import_rdd()

        config_path = rdd_path / "configs" / "default.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"RDD config not found at {config_path}")

        rdd_config = read_config_fn(str(config_path))
        rdd_config["device"] = device
        self.model = build_fn(config=rdd_config)

        if weights_path and Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"RDD weights loaded from {weights_path}")
        else:
            logger.warning(
                f"RDD weights not found at '{weights_path}'. "
                f"Using random init — download from: "
                f"https://drive.google.com/drive/folders/1QgVaqm4iTUCqbWb7_Fi6mX09EHTId0oA"
            )

        self.model.eval()
        self.model.to(device)

        # Визначаємо descriptor dim
        self._desc_dim = self._detect_desc_dim()
        logger.info(f"RDD initialized: desc_dim={self._desc_dim}, max_kpts={max_keypoints}, device={device}")

    def _detect_desc_dim(self) -> int:
        """Probe model to detect descriptor dimensionality."""
        try:
            with torch.no_grad():
                dummy = torch.randn(1, 3, 480, 640, device=self.device)
                out = self.model.extract(dummy)
                if isinstance(out, list) and len(out) > 0 and "descriptors" in out[0]:
                    return out[0]["descriptors"].shape[-1]
        except Exception as e:
            logger.warning(f"Failed to detect RDD descriptor dim: {e}. Defaulting to 128")
        return 128

    @property
    def desc_dim(self) -> int:
        return self._desc_dim

    @torch.no_grad()
    def __call__(self, input_dict: dict) -> dict:
        """ALIKED-compatible interface: input_dict = {\"image\": tensor (B,3,H,W)}.

        Returns:
            dict with \"keypoints\" (B, N, 2) and \"descriptors\" (B, N, D)
        """
        if isinstance(input_dict, dict):
            image = input_dict["image"]
        else:
            image = input_dict

        B = image.shape[0]
        all_kpts = []
        all_descs = []

        for i in range(B):
            out_list = self.model.extract(image[i:i+1])
            out = out_list[0]
            kpts = out["keypoints"]     # (1, N, 2) або (N, 2)
            descs = out["descriptors"]  # (1, N, D) або (N, D)

            # Нормалізація формату до (1, N, D)
            if kpts.dim() == 2:
                kpts = kpts.unsqueeze(0)
            if descs.dim() == 2:
                descs = descs.unsqueeze(0)

            # Обмеження кількості keypoints
            if kpts.shape[1] > self.max_keypoints:
                kpts = kpts[:, :self.max_keypoints]
                descs = descs[:, :self.max_keypoints]

            all_kpts.append(kpts)
            all_descs.append(descs)

        # Pad до однакової довжини для batch
        max_n = max(k.shape[1] for k in all_kpts)
        padded_kpts = []
        padded_descs = []
        for k, d in zip(all_kpts, all_descs):
            n = k.shape[1]
            if n < max_n:
                k = torch.cat([k, torch.zeros(1, max_n - n, 2, device=k.device)], dim=1)
                d = torch.cat([d, torch.zeros(1, max_n - n, d.shape[-1], device=d.device)], dim=1)
            padded_kpts.append(k)
            padded_descs.append(d)

        return {
            "keypoints": torch.cat(padded_kpts, dim=0),     # (B, N, 2)
            "descriptors": torch.cat(padded_descs, dim=0),  # (B, N, D)
        }

    def parameters(self):
        """Для сумісності з FeatureExtractor (перевірка is_xfeat)."""
        return self.model.parameters()
