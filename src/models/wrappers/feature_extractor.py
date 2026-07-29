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
        # Аудит §2.1: варіант із CPU-resize. Зменшення вже зроблено на numpy,
        # тож лишається сама нормалізація — Resize тут був би no-op на (S, S),
        # але зайвим ядром.
        self._dino_normalize = T.Normalize(mean=dino_mean, std=dino_std)
        self._dino_cpu_resize = bool(
            get_cfg(self.config, "models.performance.dino_cpu_resize", False)
        )
        if self._dino_cpu_resize:
            logger.info(
                f"DINO input resize on CPU ENABLED (cv2 → {dino_size}x{dino_size} "
                f"uint8 before upload). Databases built with this ON are NOT "
                f"interchangeable with those built OFF — schema fingerprint differs."
            )

        self.use_half = (
            device == "cuda"
            and torch.cuda.is_available()
            and get_cfg(self.config, "models.performance.fp16_enabled", True)
        )
        self.amp_dtype = torch.float16 if self.use_half else torch.float32

        # CPU-фолбек локалізації: на CPU-only torch device_type="cuda" в
        # autocast може впасти (а легасі torch.cuda.amp.autocast — тим паче).
        # use_half на CPU завжди False, тож autocast лишається no-op — але
        # device_type має бути коректним, інакше конструктор контексту трипить.
        self._amp_device_type = (
            "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )

        if self.use_half:
            logger.info("FP16 mixed precision ENABLED for inference")
        elif device != "cuda" or not torch.cuda.is_available():
            logger.warning(
                "Running feature extraction on CPU (no CUDA). Localization will "
                "work but be slow; build/modify the database on a GPU machine."
            )

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

    def _upload_chw(self, image: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 → (1, 3, H, W) float32 [0..1] на self.device.

        ОПТИМІЗАЦІЯ (аудит §2.1): раніше кадр конвертувався у float32 на CPU
        і вже вчетверо більшим їхав по PCIe, щоб на GPU одразу зменшитись до
        (S, S) — для DINOv3 це 224. На 1080p це ~25 МБ трансферу і ~25 МБ
        CPU-алокації на КОЖЕН форвард; при скані 4 кутів × 5 масштабів —
        до 500 МБ на keyframe.

        Тепер на девайс їде uint8 (вчетверо менше), а .float()/.div_ рахуються
        вже там. Результат ПОБІТОВО той самий: uint8→float32 точний, а ділення
        на 255.0 — одна IEEE-754 операція з тим самим округленням на CPU і CUDA.
        Тому дескриптори лишаються сумісними з уже збудованими базами.
        """
        t = torch.from_numpy(np.ascontiguousarray(image))
        t = t.permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
        return t.float().div_(255.0)

    def _cpu_resize_dino(self, image: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 → (S, S, 3) uint8 на CPU, де S = self.dino_size.

        Аудит §2.1 (повна форма). Зменшення робиться на uint8 ДО завантаження,
        тому по PCIe їде ~S²·3 байт замість H·W·3: для 1080p → 224 це ~0.15 МБ
        замість ~6.2 МБ, тобто ~40×.

        Фільтр обирається як у ResolutionNormalizer: INTER_AREA на зменшення
        (коректне усереднення площі), INTER_CUBIC на збільшення. Це НЕ той
        самий фільтр, що torchvision Resize(antialias=True), тож значення
        дескрипторів зміщуються — саме тому прапорець сидить у SCHEMA_FIELDS.

        Аспект навмисно не зберігається: цільова форма квадратна, точно як у
        ``T.Resize((S, S))``, який цей шлях заміщає. Інакше геометрія входу
        розійшлася б із GPU-варіантом.
        """
        import cv2

        s = int(self.dino_size)
        h, w = image.shape[:2]
        interp = cv2.INTER_AREA if (h > s or w > s) else cv2.INTER_CUBIC
        return cv2.resize(np.ascontiguousarray(image), (s, s), interpolation=interp)

    def _dino_input(self, image: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 → (1, 3, S, S) нормалізований тензор на self.device.

        Єдина точка препроцесу DINO. І онлайн-локалізація, і побудова бази
        ходять сюди, тож query та БД не можуть розійтися препроцесом.
        """
        if self._dino_cpu_resize:
            return self._dino_normalize(self._upload_chw(self._cpu_resize_dino(image)))
        return self.dinov2_transform(self._upload_chw(image))

    @torch.no_grad()
    def _vlad_descriptors(self, dino_input: torch.Tensor) -> np.ndarray:
        """(B, 3, S, S) → (B, out_dim) через VLAD-агрегацію патч-токенів."""
        kwargs = {}
        if self._vlad_layer is not None:
            kwargs["layer"] = self._vlad_layer
        with torch.amp.autocast(self._amp_device_type, dtype=self.amp_dtype, enabled=self.use_half):
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
            dino_input = self._dino_input(image)

        if self.vlad_aggregator is not None:
            return self._vlad_descriptors(dino_input)[0]

        if self.cesp_module is not None:
            # CESP mode: отримуємо patch tokens замість CLS
            with torch.amp.autocast(self._amp_device_type, dtype=self.amp_dtype, enabled=self.use_half):
                features = self.global_model.forward_features(dino_input)
                patch_tokens = features["x_norm_patchtokens"].float()

            # Сітка патчів — з фактичної кількості токенів, а не з хардкоду //14:
            # DINOv3 має patch_size=16 (DINOv2 — 14), і після виправлення витоку
            # register-токенів кількість токенів = (S/patch)^2 (RESEARCH 1.1).
            h_patches = w_patches = self._patch_grid_side(patch_tokens.shape[1])
            global_desc = self.cesp_module(patch_tokens, h_patches, w_patches)[0].cpu().numpy()
        else:
            # Стандартний mode: CLS token
            with torch.amp.autocast(self._amp_device_type, dtype=self.amp_dtype, enabled=self.use_half):
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
            if self._dino_cpu_resize:
                # §2.1: усі кадри стають (S, S) ще на numpy, тож батч
                # збирається одним стеком і йде на девайс ОДНИМ трансфером —
                # замість B окремих завантажень повнорозмірних кадрів.
                stacked = np.stack([self._cpu_resize_dino(img) for img in images])
                t = (
                    torch.from_numpy(np.ascontiguousarray(stacked))
                    .permute(0, 3, 1, 2)
                    .to(self.device, non_blocking=True)
                    .float()
                    .div_(255.0)
                )
                batch = self._dino_normalize(t)  # (B, 3, S, S)
            else:
                prepped = [
                    self.dinov2_transform(self._upload_chw(img))[0] for img in images
                ]
                batch = torch.stack(prepped)  # (B, 3, S, S)

            # ADDENDUM §3: чанкування батча — кап піку VRAM на слабких GPU.
            # global_batch_max=0 (дефолт) → один форвард, поведінка без змін.
            max_b = int(get_cfg(self.config, "models.performance.global_batch_max", 0) or 0)
            if max_b > 0 and batch.shape[0] > max_b:
                chunks = torch.split(batch, max_b)
            else:
                chunks = (batch,)

            outs = []
            for chunk in chunks:
                if self.vlad_aggregator is not None:
                    outs.append(np.asarray(self._vlad_descriptors(chunk)))
                elif self.cesp_module is not None:
                    with torch.amp.autocast(self._amp_device_type, dtype=self.amp_dtype, enabled=self.use_half):
                        features = self.global_model.forward_features(chunk)
                    patch_tokens = features["x_norm_patchtokens"].float()
                    h_p = w_p = self._patch_grid_side(patch_tokens.shape[1])
                    outs.append(self.cesp_module(patch_tokens, h_p, w_p).float().cpu().numpy())
                else:
                    with torch.amp.autocast(self._amp_device_type, dtype=self.amp_dtype, enabled=self.use_half):
                        outs.append(self.global_model(chunk).float().cpu().numpy())

            return np.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]

    @torch.no_grad()
    def extract_patch_tokens(self, image: np.ndarray):
        """DINO патч-токени для PCA-візуалізації (debug view «очима DINO»).

        Окремий forward саме для вікна — викликається ЛИШЕ коли вікно DINO
        відкрите (collector.want_dino_pca). Повертає (tokens, h_p, w_p), де
        tokens — (N, D) float32 на CPU, N = h_p * w_p. Той самий препроцес
        (dinov2_transform) і той самий backend (DINOv2/DINOv3), що і retrieval.
        """
        dino_input = self._dino_input(image)
        with torch.amp.autocast(self._amp_device_type, dtype=self.amp_dtype, enabled=self.use_half):
            features = self.global_model.forward_features(dino_input)
        tokens = features["x_norm_patchtokens"][0].float().cpu().numpy()  # (N, D)
        side = self._patch_grid_side(tokens.shape[0])
        return tokens, side, side

    @torch.no_grad()
    def extract_local_features(self, image: np.ndarray, static_mask: np.ndarray = None) -> dict:
        logger.debug(f"Extracting local features from image: {image.shape}")

        enhanced_image = self.preprocessor.preprocess(image)

        # Підготовка тензора (LightGlue format для ALIKED/RDD; сирий (1,3,H,W) для XFeat).
        # §2.1: uint8 на девайс, float/div — уже там (той самий результат, 4× менше PCIe).
        rgb_tensor = self._upload_chw(enhanced_image)

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

        # XFeat має інший інтерфейс (detectAndCompute на сирому тензорі), ніж
        # ALIKED/RDD (виклик як {"image": tensor}). Ця гілка дзеркалить
        # batch-шлях extract_features_batch, щоб онлайн-локалізація давала той
        # самий формат ознак, що й БД, збудована XFeat-ом.
        is_xfeat = "XFeat" in self.local_model.__class__.__name__

        with Telemetry.profile("local_extractor"):
            if is_xfeat:
                top_k = get_cfg(self.config, "models.xfeat.top_k", 2048)
                xf = self.local_model.detectAndCompute(rgb_tensor, top_k=top_k)[0]
                keypoints = xf["keypoints"].cpu().numpy()
                descriptors = xf["descriptors"].cpu().numpy()
            else:
                # ALIKED нестабільний усередині AMP autocast (NaN) — тримаємо FP32.
                with contextlib.nullcontext():
                    aliked_out = self.local_model({"image": rgb_tensor})
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
                # ВИПРАВЛЕНО: тут було len(aliked_out[...]), а aliked_out існує
                # лише в ALIKED/RDD-гілці — на XFeat це UnboundLocalError у
                # момент, коли маска зрізала все. keypoints у скоупі завжди.
                logger.warning(
                    f"All keypoints filtered out by YOLO mask! "
                    f"Image {image.shape[:2]}, total_kpts={len(keypoints)}, "
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
        # Аудит §2.1/§2.4: раніше кожне зображення окремо йшло через
        # torch.tensor(..., pin_memory=True).float() — тобто (а) копія + власна
        # pinned-алокація на КОЖЕН кадр (cudaHostAlloc синхронізує драйвер), і
        # (б) float32 їхав по PCIe вчетверо більшим за потрібне. Тепер батч
        # збирається як uint8 одним numpy-стеком, а .float()/.div_ рахуються
        # на девайсі. Числовий результат той самий.
        # §2.1: коли CPU-resize увімкнено, зменшуємо ДО стеку — тоді на девайс
        # їде (B, 3, S, S) замість (B, 3, H, W). Це ТОЙ САМИЙ препроцес, що в
        # _dino_input на онлайн-шляху: інакше дескриптори БД і запиту були б
        # порахованими різними фільтрами.
        _dino_src = (
            [self._cpu_resize_dino(img) for img in images]
            if self._dino_cpu_resize
            else images
        )
        dino_batch = (
            torch.from_numpy(np.ascontiguousarray(np.stack(_dino_src)))
            .permute(0, 3, 1, 2)
            .to(self.device, non_blocking=True)
            .float()
            .div_(255.0)
        )
        dino_input = (
            self._dino_normalize(dino_batch)
            if self._dino_cpu_resize
            else self.dinov2_transform(dino_batch)
        )

        # 2. Prepare Local Tensor
        prep_images = [self.preprocessor.preprocess(img) for img in images]
        local_batch = (
            torch.from_numpy(np.ascontiguousarray(np.stack(prep_images)))
            .permute(0, 3, 1, 2)
            .to(self.device, non_blocking=True)
            .float()
            .div_(255.0)
        )

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
        # Аудит §2.4: dino_input і local_batch створюються на DEFAULT-стрімі, а
        # споживаються на бічних. Без wait_stream ядра бічного стріму можуть
        # стартувати ДО завершення підготовки — гонка, що проявляється рідким
        # NaN/сміттям, а не падінням. record_stream нижче не дає кешуючому
        # алокатору переюзати ці блоки, поки бічні стріми з них читають.
        if self.device == "cuda":
            current = torch.cuda.current_stream()
            for s in (stream_global, stream_local):
                if s is not None:
                    s.wait_stream(current)
            for t, s in ((dino_input, stream_global), (local_batch, stream_local)):
                if s is not None:
                    t.record_stream(s)

        context_global = (
            torch.cuda.stream(stream_global) if stream_global else contextlib.nullcontext()
        )
        with context_global:
            with Telemetry.profile("dinov2"):
                if self.vlad_aggregator is not None:
                    out_global = torch.from_numpy(self._vlad_descriptors(dino_input))
                elif self.cesp_module is not None:
                    with torch.amp.autocast(self._amp_device_type, dtype=self.amp_dtype, enabled=self.use_half):
                        features = self.global_model.forward_features(dino_input)
                    patch_tokens = features["x_norm_patchtokens"].float()
                    # RESEARCH 1.1: сітка з фактичної кількості токенів, не //14
                    h_p = w_p = self._patch_grid_side(patch_tokens.shape[1])
                    out_global = self.cesp_module(patch_tokens, h_p, w_p)
                else:
                    with torch.amp.autocast(self._amp_device_type, dtype=self.amp_dtype, enabled=self.use_half):
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
                    # ВИПРАВЛЕНО: розмірність дескриптора беремо з самого масиву,
                    # а не з хардкоду 128 (ALIKED=128, XFeat=64, RDD=256) —
                    # інакше порожній результат мав чужу ширину.
                    desc_dim = desc.shape[1] if desc.ndim == 2 else 128
                    kp = np.empty((0, 2), dtype=np.float32)
                    desc = np.empty((0, desc_dim), dtype=np.float32)

            results.append({
                "keypoints": kp, "descriptors": desc, "coords_2d": kp.copy(), "global_desc": gd,
                "image_size": np.array([images[i].shape[0], images[i].shape[1]], dtype=np.int32),
            })

        return results
