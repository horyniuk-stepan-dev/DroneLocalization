import contextlib
import numpy as np
import torch
import torchvision.transforms as T

from config.config import get_cfg
from src.utils.image_preprocessor import ImagePreprocessor
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """Combined feature extraction (ALIKED + DINOv2 [+ CESP])"""

    def __init__(self, local_model, global_model, device="cuda", config=None, cesp_module=None):
        self.local_model = local_model  # ALIKED
        self.global_model = global_model  # DINOv2
        self.device = device
        self.config = config or {}
        self.preprocessor = ImagePreprocessor(config)
        self.cesp_module = cesp_module  # Опціональний CESP для покращення global descriptors

        # Трансформації для DINOv2 (ImageNet стандарти)
        dino_size = get_cfg(self.config, "dinov2.input_size", 336)
        self.dino_size = dino_size
        self.dinov2_transform = T.Compose(
            [
                T.Resize((dino_size, dino_size), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        logger.info(
            f"FeatureExtractor initialized with ALIKED and DINOv2 ({cesp_status}) on {device}"
        )
        
        if device == "cuda":
            self.stream_global = torch.cuda.Stream()
            self.stream_local = torch.cuda.Stream()
        else:
            self.stream_global = None
            self.stream_local = None

    @torch.no_grad()
    def extract_global_descriptor(self, image: np.ndarray) -> np.ndarray:
        logger.debug("Extracting global descriptor with DINOv2...")
        dino_tensor = torch.from_numpy(image).float().div_(255.0)
        dino_tensor = dino_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
        dino_input = self.dinov2_transform(dino_tensor)

        if self.cesp_module is not None:
            # CESP mode: отримуємо patch tokens замість CLS
            with torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=self.use_half):
                features = self.global_model.forward_features(dino_input)
                patch_tokens = features["x_norm_patchtokens"].float()

            h_patches = self.dino_size // 14
            w_patches = self.dino_size // 14
            global_desc = self.cesp_module(patch_tokens, h_patches, w_patches)[0].cpu().numpy()
        else:
            # Стандартний mode: CLS token
            with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_half):
                global_desc = self.global_model(dino_input)[0].float().cpu().numpy()

        return global_desc

    @torch.no_grad()
    def extract_local_features(self, image: np.ndarray, static_mask: np.ndarray = None) -> dict:
        logger.debug(f"Extracting local features (ALIKED) from image: {image.shape}")

        enhanced_image = self.preprocessor.preprocess(image)

        # Підготовка зображення для ALIKED (LightGlue format)
        rgb_tensor = torch.from_numpy(enhanced_image).float().div_(255.0)
        rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)

        # ALIKED очікує словник зі списком/тензором 'image'
        input_dict = {"image": rgb_tensor}

        # ALIKED behaves unstably and yields NaNs inside AMP autocast. Always run it in FP32!
        with contextlib.nullcontext():
            aliked_out = self.local_model(input_dict)

        # LightGlue ALIKED wrapper повертає батч: (1, N, 2) та (1, N, 128)
        keypoints = aliked_out["keypoints"][0].cpu().numpy()
        descriptors = aliked_out["descriptors"][0].cpu().numpy()

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
                logger.warning("All keypoints filtered out by YOLO mask!")

        return {"keypoints": keypoints, "descriptors": descriptors, "coords_2d": keypoints.copy()}

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
    def extract_features_batch(self, images: list[np.ndarray], static_masks: list[np.ndarray]) -> list[dict]:
        """
        Extracts features for a batch of images using CUDA streams for parallel execution.
        """
        B = len(images)
        if B == 0:
            return []

        # 1. Prepare DINOv2 Tensor
        dino_tensors = []
        for img in images:
            rgb = torch.from_numpy(img).float().div_(255.0)
            dino_tensors.append(rgb.permute(2, 0, 1))
        dino_batch = torch.stack(dino_tensors).to(self.device, non_blocking=True)
        dino_input = self.dinov2_transform(dino_batch)

        # 2. Prepare ALIKED Tensor
        prep_images = [self.preprocessor.preprocess(img) for img in images]
        aliked_tensors = []
        for p_img in prep_images:
            rgb = torch.from_numpy(p_img).float().div_(255.0)
            aliked_tensors.append(rgb.permute(2, 0, 1))
        aliked_batch = torch.stack(aliked_tensors).to(self.device, non_blocking=True)
        input_dict = {"image": aliked_batch}

        stream_global = self.stream_global if self.device == "cuda" else None
        stream_local = self.stream_local if self.device == "cuda" else None

        global_descs = None
        aliked_out = None

        # PARALLEL EXECUTION
        context_global = torch.cuda.stream(stream_global) if stream_global else contextlib.nullcontext()
        with context_global:
            if self.cesp_module is not None:
                with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_half):
                    features = self.global_model.forward_features(dino_input)
                    patch_tokens = features["x_norm_patchtokens"].float()
                h_p, w_p = self.dino_size // 14, self.dino_size // 14
                out_global = self.cesp_module(patch_tokens, h_p, w_p)
            else:
                with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_half):
                    out_global = self.global_model(dino_input).float()

        out_kpts = []
        out_descs = []
        context_local = torch.cuda.stream(stream_local) if stream_local else contextlib.nullcontext()
        with context_local:
            for b in range(B):
                single_img = aliked_batch[b:b+1]  # shape (1, 3, H, W)
                input_dict = {"image": single_img}
                # ALIKED behaves unstably and yields NaNs inside AMP autocast. Always run it in FP32!
                aliked_out = self.local_model(input_dict)
                out_kpts.append(aliked_out["keypoints"][0].float())
                out_descs.append(aliked_out["descriptors"][0].float())

        if self.device == "cuda":
            torch.cuda.synchronize()

        global_descs = out_global.cpu().numpy()
        keypoints_batch = [kp.cpu().numpy() for kp in out_kpts]
        descriptors_batch = [desc.cpu().numpy() for desc in out_descs]

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
                "keypoints": kp,
                "descriptors": desc,
                "coords_2d": kp.copy(),
                "global_desc": gd
            })

        return results
