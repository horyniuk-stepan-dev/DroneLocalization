import contextlib
import math
import os

import numpy as np
import torch
import torchvision.transforms as T

from config import get_active_descriptor_cfg, get_cfg
from src.utils.image_preprocessor import ImagePreprocessor
from src.utils.logging_utils import get_logger
from src.utils.telemetry import Telemetry

logger = get_logger(__name__)


class FeatureExtractor:
    """Combined feature extraction (ALIKED/RDD + DINOv2 [+ CESP])"""

    def __init__(self, local_model, global_model, device="cuda", config=None, cesp_module=None):
        self.local_model = local_model  # ALIKED або RDD
        self.global_model = global_model  # DINOv2
        self.device = device
        self.config = config or {}
        self.preprocessor = ImagePreprocessor(config)
        self.cesp_module = cesp_module  # Опціональний CESP для покращення global descriptors

        # ── RESEARCH 2.1 (AnyLoc): VLAD-агрегація патч-токенів ──────────────
        # Вантажиться з конфігу тут (а не в місцях конструювання), щоб усі
        # 4 точки створення FeatureExtractor отримали її автоматично.
        self.vlad_aggregator = None
        self._vlad_layer = None
        if get_cfg(config, "models.vlad.enabled", False):
            vocab_path = get_cfg(config, "models.vlad.vocab_path", None)
            if vocab_path and os.path.exists(vocab_path):
                from src.models.wrappers.vlad_aggregator import VladAggregator

                self.vlad_aggregator = VladAggregator.load(
                    vocab_path,
                    low_norm_fraction=get_cfg(config, "models.vlad.low_norm_fraction", 0.0),
                )
                self._vlad_layer = get_cfg(config, "models.vlad.layer", None)
                if cesp_module is not None:
                    logger.warning("Both VLAD and CESP enabled — VLAD takes precedence")
            else:
                logger.warning(
                    f"models.vlad.enabled=True but vocab_path is missing or not found "
                    f"({vocab_path!r}) — falling back to CLS. "
                    f"Build the vocabulary with scripts/build_vlad_vocab.py"
                )

        # Параметри нормалізації та розміру входу — беремо з активного backend (dinov2 або dinov3)
        _desc_cfg = get_active_descriptor_cfg(self.config)
        dino_size = _desc_cfg.input_size
        dino_mean = _desc_cfg.normalize_mean
        dino_std = _desc_cfg.normalize_std
        self.dino_size = dino_size
        self.dinov2_transform = T.Compose(
            [
                T.Resize((dino_size, dino_size), antialias=True),
                T.Normalize(mean=dino_mean, std=dino_std),
            ]
        )

        self.use_half = (
            device == "cuda"
            and torch.cuda.is_available()
            and get_cfg(self.config, "models.performance.fp16_enabled", True)
        )
        self.amp_dtype = torch.float16 if self.use_half else torch.float32

        if self.use_half:
            logger.info("FP16 mixed precision ENABLED for inference")

        cesp_status = "with CESP" if cesp_module is not None else "without CESP"
        local_name = type(local_model).__name__
        global_name = type(global_model).__name__
        logger.info(
            f"FeatureExtractor initialized: local={local_name}, global={global_name} "
            f"({cesp_status}) | input_size={dino_size}, mean={dino_mean}, std={dino_std} "
            f"| device={device}"
        )

        if device == "cuda":
            self.stream_global = torch.cuda.Stream()
            self.stream_local = torch.cuda.Stream()
        else:
            self.stream_global = None
            self.stream_local = None

    @staticmethod
    def _patch_grid_side(n_tokens: int) -> int:
        """Сторона квадратної сітки патчів із кількості патч-токенів.

        Вхід DINO — квадрат (S, S), тож токенів має бути side². Якщо ні —
        у токени протекли register-токени або вхід не квадратний; беремо
        floor(sqrt) і попереджаємо, щоб CESP не отримав неузгоджену сітку.
        """
        side = int(math.isqrt(int(n_tokens)))
        if side * side != int(n_tokens):
            logger.warning(
                f"Patch tokens count {n_tokens} is not a perfect square — "
                f"possible register-token leak or non-square input; using side={side}"
            )
        return side

    @property
    def global_descriptor_dim(self) -> int:
        """Фактична розмірність глобального дескриптора (VLAD змінює її)."""
        if self.vlad_aggregator is not None:
            return self.vlad_aggregator.out_dim
        return get_active_descriptor_cfg(self.config).descriptor_dim

    @torch.no_grad()
    def _vlad_descriptors(self, dino_input: torch.Tensor) -> np.ndarray:
        """(B, 3, S, S) → (B, out_dim) через VLAD-агрегацію патч-токенів."""
        kwargs = {}
        if self._vlad_layer is not None:
            kwargs["layer"] = self._vlad_layer
        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_half):
            try:
                features = self.global_model.forward_features(dino_input, **kwargs)
            except TypeError:
                # DINOv2 (torch.hub) не приймає layer — беремо останній шар
                features = self.global_model.forward_features(dino_input)
        tokens = features["x_norm_patchtokens"].float().cpu().numpy()
        return self.vlad_aggregator.aggregate_batch(tokens)

    @torch.no_grad()
    def extract_global_descriptor(self, image: np.ndarray) -> np.ndarray:
        with Telemetry.profile("dinov2"):
            logger.debug("Extracting global descriptor with DINOv2...")
            dino_tensor = torch.from_numpy(image).float().div_(255.0)
            dino_tensor = (
                dino_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
            )
            dino_input = self.dinov2_transform(dino_tensor)

        if self.vlad_aggregator is not None:
            return self._vlad_descriptors(dino_input)[0]

        if self.cesp_module is not None:
            # CESP mode: отримуємо patch tokens замість CLS
            with torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=self.use_half):
                features = self.global_model.forward_features(dino_input)
                patch_tokens = features["x_norm_patchtokens"].float()

            # Сітка патчів — з фактичної кількості токенів, а не з хардкоду //14:
            # DINOv3 має patch_size=16 (DINOv2 — 14), і після виправлення витоку
            # register-токенів кількість токенів = (S/patch)^2 (RESEARCH 1.1).
            h_patches = w_patches = self._patch_grid_side(patch_tokens.shape[1])
            global_desc = self.cesp_module(patch_tokens, h_patches, w_patches)[0].cpu().numpy()
        else:
            # Стандартний mode: CLS token
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_half):
                global_desc = self.global_model(dino_input)[0].float().cpu().numpy()

        return global_desc

    @torch.no_grad()
    def extract_global_descriptors_multi(self, images: list[np.ndarray]) -> np.ndarray:
        """Глобальні дескриптори для СПИСКУ зображень одним forward-пасом.

        A2: використовується для 4 ротацій кадру при auto_rotation — один
        батчований ViT-forward замість чотирьох послідовних (~3× швидше на GPU).
        Зображення можуть мати різні розміри (90°-ротації), тому resize
        виконується по-кадрово, а батчується вже (B, 3, S, S).
        """
        if not images:
            return np.empty((0, 0), dtype=np.float32)

        with Telemetry.profile("dinov2"):
            prepped = []
            for img in images:
                t = torch.from_numpy(np.ascontiguousarray(img)).float().div_(255.0)
                t = t.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
                prepped.append(self.dinov2_transform(t)[0])
            batch = torch.stack(prepped)  # (B, 3, S, S)

            if self.vlad_aggregator is not None:
                return self._vlad_descriptors(batch)

            if self.cesp_module is not None:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_half):
                    features = self.global_model.forward_features(batch)
                patch_tokens = features["x_norm_patchtokens"].float()
                h_p = w_p = self._patch_grid_side(patch_tokens.shape[1])
                out = self.cesp_module(patch_tokens, h_p, w_p)
            else:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_half):
                    out = self.global_model(batch)

            return out.float().cpu().numpy()

    @torch.no_grad()
    def extract_patch_tokens(self, image: np.ndarray):
        """DINO патч-токени для PCA-візуалізації (debug view «очима DINO»).

        Окремий forward саме для вікна — викликається ЛИШЕ коли вікно DINO
        відкрите (collector.want_dino_pca). Повертає (tokens, h_p, w_p), де
        tokens — (N, D) float32 на CPU, N = h_p * w_p. Той самий препроцес
        (dinov2_transform) і той самий backend (DINOv2/DINOv3), що і retrieval.
        """
        dino_tensor = torch.from_numpy(np.ascontiguousarray(image)).float().div_(255.0)
        dino_tensor = (
            dino_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
        )
        dino_input = self.dinov2_transform(dino_tensor)
        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_half):
            features = self.global_model.forward_features(dino_input)
        tokens = features["x_norm_patchtokens"][0].float().cpu().numpy()  # (N, D)
        side = self._patch_grid_side(tokens.shape[0])
        return tokens, side, side

    @torch.no_grad()
    def extract_local_features(self, image: np.ndarray, static_mask: np.ndarray = None) -> dict:
        logger.debug(f"Extracting local features (ALIKED) from image: {image.shape}")

        enhanced_image = self.preprocessor.preprocess(image)

        # Підготовка зображення для ALIKED (LightGlue format)
        rgb_tensor = torch.from_numpy(enhanced_image).float().div_(255.0)
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)

        # Fix OOM: Downscale high-resolution frames (e.g. 4K) to prevent massive memory spikes
        max_edge = get_cfg(self.config, "localization.max_local_edge", 1600)
        orig_h, orig_w = rgb_tensor.shape[2], rgb_tensor.shape[3]
        scale_factor = 1.0
        if max(orig_h, orig_w) > max_edge:
            scale_factor = max_edge / float(max(orig_h, orig_w))
            new_h, new_w = int(orig_h * scale_factor), int(orig_w * scale_factor)
            rgb_tensor = torch.nn.functional.interpolate(
                rgb_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False
            )
            logger.debug(f"Downscaled local extraction from {orig_w}x{orig_h} to {new_w}x{new_h}")

        # ALIKED очікує словник зі списком/тензором 'image'
        input_dict = {"image": rgb_tensor}

        # ALIKED behaves unstably and yields NaNs inside AMP autocast. Always run it in FP32!
        with Telemetry.profile("local_extractor"):
            with contextlib.nullcontext():
                aliked_out = self.local_model(input_dict)

        # LightGlue wrapper повертає батч: (1, N, 2) та (1, N, D)
        keypoints = aliked_out["keypoints"][0].cpu().numpy()
        descriptors = aliked_out["descriptors"][0].cpu().numpy()

        if scale_factor != 1.0:
            keypoints = keypoints / scale_factor

        # Фільтрація точок за маскою динамічних об'єктів (YOLO)
        if static_mask is not None and len(keypoints) > 0:
            # Vectorized YOLO mask filtering
            ix = np.round(keypoints[:, 0]).astype(np.intp)
            iy = np.round(keypoints[:, 1]).astype(np.intp)
            in_bounds = (
                (iy >= 0) & (iy < static_mask.shape[0]) & (ix >= 0) & (ix < static_mask.shape[1])
            )
            valid = np.zeros(len(keypoints), dtype=bool)
            valid[in_bounds] = static_mask[iy[in_bounds], ix[in_bounds]] > 128

            if valid.any():
                keypoints = keypoints[valid]
                descriptors = descriptors[valid]
            else:
                logger.warning(
                    f"All keypoints filtered out by YOLO mask! "
                    f"Image {image.shape[:2]}, total_kpts={len(aliked_out['keypoints'][0])}, "
                    f"mask_static_ratio={np.mean(static_mask > 128):.1%}. "
                    f"The entire image may be covered by dynamic objects (vehicles, people)."
                )

        return {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "coords_2d": keypoints.copy(),
            "image_size": np.array([image.shape[0], image.shape[1]], dtype=np.int32),
        }

    @torch.no_grad()
    def extract_features(self, image: np.ndarray, static_mask: np.ndarray = None) -> dict:
        local_feats = self.extract_local_features(image, static_mask)
        global_desc = self.extract_global_descriptor(image)
        local_feats["global_desc"] = global_desc

        # logger.success(
        #     f"Extracted {len(local_feats['keypoints'])} ALIKED keypoints, global DINOv2 desc dim {len(global_desc)}"
        # )
        return local_feats

    @torch.no_grad()
    def extract_features_batch(
        self, images: list[np.ndarray], static_masks: list[np.ndarray]
    ) -> list[dict]:
        """
        Extracts features for a batch of images using CUDA streams for parallel execution.
        """
        B = len(images)
        if B == 0:
            return []

        # 1. Prepare DINOv2 Tensor
        dino_tensors = []
        for img in images:
            rgb = torch.tensor(img, pin_memory=True).float().div_(255.0)
            dino_tensors.append(rgb.permute(2, 0, 1))
        dino_batch = torch.stack(dino_tensors).to(self.device, non_blocking=True)
        dino_input = self.dinov2_transform(dino_batch)

        # 2. Prepare Local Tensor
        prep_images = [self.preprocessor.preprocess(img) for img in images]
        local_tensors = []
        for p_img in prep_images:
            rgb = torch.tensor(p_img, pin_memory=True).float().div_(255.0)
            local_tensors.append(rgb.permute(2, 0, 1))
        local_batch = torch.stack(local_tensors).to(self.device, non_blocking=True)

        # Fix OOM: Downscale high-resolution frames (e.g. 4K) to prevent massive memory spikes
        max_edge = get_cfg(self.config, "localization.max_local_edge", 1600)
        orig_h, orig_w = local_batch.shape[2], local_batch.shape[3]
        scale_factor = 1.0
        if max(orig_h, orig_w) > max_edge:
            scale_factor = max_edge / float(max(orig_h, orig_w))
            new_h, new_w = int(orig_h * scale_factor), int(orig_w * scale_factor)
            local_batch = torch.nn.functional.interpolate(
                local_batch, size=(new_h, new_w), mode='bilinear', align_corners=False
            )
            logger.debug(f"Downscaled local batch extraction from {orig_w}x{orig_h} to {new_w}x{new_h}")

        is_xfeat = (
            hasattr(self.local_model, "__class__")
            and "XFeat" in self.local_model.__class__.__name__
        )
        input_dict = {"image": local_batch} if not is_xfeat else local_batch

        stream_global = self.stream_global if self.device == "cuda" else None
        stream_local = self.stream_local if self.device == "cuda" else None

        global_descs = None
        aliked_out = None

        # PARALLEL EXECUTION
        context_global = (
            torch.cuda.stream(stream_global) if stream_global else contextlib.nullcontext()
        )
        with context_global:
            with Telemetry.profile("dinov2"):
                if self.vlad_aggregator is not None:
                    out_global = torch.from_numpy(self._vlad_descriptors(dino_input))
                elif self.cesp_module is not None:
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_half):
                        features = self.global_model.forward_features(dino_input)
                    patch_tokens = features["x_norm_patchtokens"].float()
                    # RESEARCH 1.1: сітка з фактичної кількості токенів, не //14
                    h_p = w_p = self._patch_grid_side(patch_tokens.shape[1])
                    out_global = self.cesp_module(patch_tokens, h_p, w_p)
                else:
                    with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_half):
                        out_global = self.global_model(dino_input).float()

        out_kpts = []
        out_descs = []
        context_local = (
            torch.cuda.stream(stream_local) if stream_local else contextlib.nullcontext()
        )
        with context_local:
            with Telemetry.profile("local_extractor"):
                if is_xfeat:
                    # S3-1: Native True Batching for XFeat
                    xfeat_out = self.local_model.detectAndCompute(
                        input_dict, top_k=get_cfg(self.config, "models.xfeat.top_k", 2048)
                    )
                    for res in xfeat_out:
                        out_kpts.append(res["keypoints"].float())
                        out_descs.append(res["descriptors"].float())
                else:
                    # S3-1: ALIKED fallback. Unstable inside true batch, iterating frames natively.
                    for b in range(B):
                        single_img = local_batch[b : b + 1]  # shape (1, 3, H, W)
                        aliked_in = {"image": single_img}
                        aliked_out = self.local_model(aliked_in)
                        out_kpts.append(aliked_out["keypoints"][0].float())
                        out_descs.append(aliked_out["descriptors"][0].float())

        if self.device == "cuda":
            torch.cuda.synchronize()

        global_descs = out_global.cpu().numpy()
        keypoints_batch = [kp.cpu().numpy() for kp in out_kpts]
        descriptors_batch = [desc.cpu().numpy() for desc in out_descs]

        if scale_factor != 1.0:
            keypoints_batch = [kp / scale_factor for kp in keypoints_batch]

        # Assembly
        results = []
        for i in range(B):
            kp = keypoints_batch[i]
            desc = descriptors_batch[i]
            mask = static_masks[i]
            gd = global_descs[i]

            if mask is not None and len(kp) > 0:
                ix = np.round(kp[:, 0]).astype(np.intp)
                iy = np.round(kp[:, 1]).astype(np.intp)
                in_bounds = (iy >= 0) & (iy < mask.shape[0]) & (ix >= 0) & (ix < mask.shape[1])
                valid = np.zeros(len(kp), dtype=bool)
                valid[in_bounds] = mask[iy[in_bounds], ix[in_bounds]] > 128

                if valid.any():
                    kp = kp[valid]
                    desc = desc[valid]
                else:
                    kp = np.empty((0, 2), dtype=np.float32)
                    desc = np.empty((0, 128), dtype=np.float32)

            results.append({
                "keypoints": kp, "descriptors": desc, "coords_2d": kp.copy(), "global_desc": gd,
                "image_size": np.array([images[i].shape[0], images[i].shape[1]], dtype=np.int32),
            })

        return results
