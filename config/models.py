"""Model configuration: DINOv2/v3, YOLO, local extractors, VRAM, performance."""

from pydantic import BaseModel, Field


class Dinov2ModelConfig(BaseModel):
    """DINOv2 ViT-L/14 — ImageNet pretrained, завантажується через torch.hub"""

    descriptor_dim: int = 1024
    input_size: int = 336
    normalize_mean: list[float] = [0.485, 0.456, 0.406]
    normalize_std: list[float] = [0.229, 0.224, 0.225]
    hub_repo: str = "facebookresearch/dinov2"
    hub_model: str = "dinov2_vitl14"
    vram_required_mb: float = 1600.0


class Dinov3ModelConfig(BaseModel):
    """DINOv3 ViT-L/16 — pretrained на 493M супутникових знімків, HuggingFace"""

    descriptor_dim: int = 1024
    input_size: int = 224
    normalize_mean: list[float] = [0.430, 0.411, 0.296]
    normalize_std: list[float] = [0.213, 0.156, 0.143]
    hf_model_id: str = "facebook/dinov3-vitl16-pretrain-sat493m"
    # БЕЗПЕКА: модель вантажиться з trust_remote_code=True. Зафіксуйте commit
    # hash репозиторію HF, щоб підміна upstream-коду не стала виконанням
    # чужого коду на вашій машині. Порожньо = latest (з warning у лог).
    hf_revision: str = ""
    vram_required_mb: float = 1600.0


class GlobalDescriptorConfig(BaseModel):
    """Вибір глобального дескриптора: 'dinov2' або 'dinov3'"""

    backend: str = "dinov3"  # "dinov2" | "dinov3"
    dinov2: Dinov2ModelConfig = Dinov2ModelConfig()
    dinov3: Dinov3ModelConfig = Dinov3ModelConfig()

    def active(self) -> Dinov2ModelConfig | Dinov3ModelConfig:
        """Повертає конфіг активної моделі."""
        return self.dinov2 if self.backend == "dinov2" else self.dinov3


class YoloConfig(BaseModel):
    """YOLOv11n-seg (Nano) for dynamic object masking."""

    model_path: str = "models/yolo11n-seg.pt"
    vram_required_mb: float = 200.0


class DepthEstimatorConfig(BaseModel):
    """Monocular depth during the database build (metadata/depth_scales) and for
    localization.scale_use_depth_hint. "none" skips the model entirely."""

    backend: str = "depth_anything_v2"  # "depth_anything_v2" | "none"


# ── Local feature models ───────────────────────────────────────────────────────
# One schema per model: every field below is passed to that model's loader.
# (Until 2026-10 all extractors and matchers shared one generic ModelSettings,
# so most of its fields — nms_radius / detection_threshold for ALIKED,
# depth/width_confidence for LightGlue, dtype, hub_* for local models — were
# written to user_config.json but never reached the model.)
#
# Defaults equal the values the code ACTUALLY ran with before the split
# (the lightglue library defaults), so moving to these classes changes nothing.


class AlikedSettings(BaseModel):
    """ALIKED (lightglue.ALIKED). Feeds both the database build and localization."""

    model_name: str = "aliked-n16"
    max_keypoints: int = 4096
    # > 0: threshold mode — keypoints with score > threshold, capped at
    # max_keypoints. <= 0: exact top-k (always max_keypoints points).
    # Changing it changes the keypoint set: rebuild databases for a fair A/B.
    detection_threshold: float = 0.2
    nms_radius: int = 2
    vram_required_mb: float = 400.0


class XFeatSettings(BaseModel):
    hub_repo: str = "verlab/accelerated_features"
    hub_model: str = "XFeat"
    top_k: int = 2048
    # quality_preset of forks that support it ("fast" | ...); ignored otherwise.
    preset: str = "fast"
    vram_required_mb: float = 300.0


class SuperPointSettings(BaseModel):
    max_keypoints: int = 4096
    nms_radius: int = 4
    detection_threshold: float = 0.0005
    vram_required_mb: float = 500.0


class RddSettings(BaseModel):
    model_path: str = "models/RDD-v2.pth"
    max_keypoints: int = 4096
    vram_required_mb: float = 500.0


class LightGlueSettings(BaseModel):
    """LightGlue matcher (one block per feature type)."""

    model_path: str | None = ""
    backend: str = "git"  # "git" | "torchscript" | "tensorrt"
    auto_convert: bool = False
    vram_required_mb: float = 800.0
    # Adaptive early exit (lightglue: depth_confidence). -1 disables it.
    depth_confidence: float = 0.95
    # Point pruning (lightglue: width_confidence); active on CUDA with
    # >= 1024 keypoints. -1 disables it.
    width_confidence: float = 0.99
    # Matches with assignment score below this are dropped.
    filter_threshold: float = 0.1
    flash: bool = True  # FlashAttention when available
    mixed_precision: bool = False  # lightglue "mp"


class CespConfig(BaseModel):
    enabled: bool = False
    weights_path: str | None = None
    scales: list[int] = [1, 2, 4]


class VladConfig(BaseModel):
    """RESEARCH 2.1 (AnyLoc): ненавчена VLAD-агрегація патч-токенів DINOv3.

    enabled=True вимагає vocab_path — словник, збудований
    scripts/build_vlad_vocab.py на референсних кадрах ТОГО САМОГО домену.
    База даних має бути перебудована з тим самим словником (розмірність
    глобального дескриптора змінюється: 1024 → pca_dim).
    """

    enabled: bool = False
    vocab_path: str | None = None
    n_clusters: int = 32
    pca_dim: int = 512
    # Проміжний шар ViT для патч-токенів (None = останній, з фінальним
    # LayerNorm). AnyLoc показує кращі результати з проміжних шарів —
    # підбирати валідацією, значення залежить від бекбона.
    layer: int | None = None
    # Dustbin-сурогат (SALAD): відкинути цю частку патчів з найнижчою
    # L2-нормою токена перед агрегацією (0.0 = вимкнено, 0.1 = 10%).
    low_norm_fraction: float = 0.0


class VramManagementConfig(BaseModel):
    max_vram_ratio: float = 0.8
    default_required_mb: float = 2000.0


class ModelsCacheConfig(BaseModel):
    engine_cache_dir: str = "models/engines/"
    # auto_compile REMOVED (2026-10): nothing compiled engines on demand; build
    # them with scripts/compile_dinov2_trt.py.


class PerformanceConfig(BaseModel):
    auto_tune: bool = True  # Auto-detect hardware and tune batch sizes, threads, VRAM limits
    # auto_tune_vram_headroom and propagation_max_workers REMOVED (2026-10):
    # nothing read them — propagation matches pairs sequentially on one GPU.
    fp16_enabled: bool = True
    # ADDENDUM §3 (слабкі GPU): максимальний батч ViT-форварда в
    # extract_global_descriptors_multi (recovery: до 20 кадрів разом).
    # 0 = без ліміту (ПОТОЧНА поведінка). На 4 GB VRAM безпечно 4-6.
    # Впливає лише на пік памʼяті; дескриптори побітово ті самі.
    global_batch_max: int = 0

    # ── Аудит §2.1 (повна форма): зменшувати кадр до входу DINO на CPU ───────
    # Зараз кадр цілком їде на GPU, щоб там же зменшитись до (S, S) — для
    # DINOv3 це 224. На 1080p це ~6.2 МБ uint8 замість ~0.15 МБ, тобто ~40×
    # зайвого PCIe-трафіку на КОЖЕН форвард; при скані 4 кутів × 5 масштабів
    # різниця набігає в сотні мегабайт на keyframe.
    #
    # УВАГА: cv2.INTER_AREA і torchvision Resize(antialias) — РІЗНІ фільтри,
    # тож значення дескрипторів зміщуються. Тому ключ входить у
    # SCHEMA_FIELDS (src/database/schema_fingerprint.py): база, збудована з
    # іншим значенням, детектується як несумісна замість тихого псування
    # матчів. З цієї ж причини його НЕМАЄ в hardware_profile.TUNABLE_KEYS —
    # auto_tune не має права робити бази машинозалежними.
    #
    # Дефолт False = ПОТОЧНА поведінка (resize на GPU).
    dino_cpu_resize: bool = False

    torch_compile: bool = False
    use_tensorrt_for_yolo: bool = False  # portable across GPUs; TRT engines are hardware-specific
    log_level: str = "INFO"
    debug_mode: bool = True
    # HARDENING P0-5: коли True — сирі lat/lon та координати якорів у логах
    # округлюються/маскуються (щоб захоплений app.log не видавав маршрут місії).
    # Дефолт False = ПОТОЧНА поведінка (повна точність). Вмикається у
    # user_config.json для польових збірок.
    redact_coords_in_logs: bool = False

    # HARDENING P1-8 (safe slice): детермінований режим для обмеження
    # найгіршої латентності. True вимикає cudnn.benchmark (прибирає змінну
    # вартість першого виклику та недетермінований вибір ядер). Дефолт
    # False = ПОТОЧНА поведінка (benchmark=True, кращий throughput).
    deterministic: bool = False
    # Логувати перцентилі per-frame латентності (p50/p95/p99/max) кожні
    # latency_log_interval кадрів. Дефолт off = поточна поведінка.
    log_latency_stats: bool = False
    latency_log_interval: int = 100

    # HARDENING P2-12: перевірка цілісності ваг на старті проти закріпленого
    # SHA-256 маніфесту (models/weights_manifest.json). "off" (дефолт) =
    # поточна поведінка; "warn" = логувати розбіжності; "enforce" = аварійно
    # завершити старт, якщо файл ваг відсутній/змінений (захист від підміни).
    weight_integrity_mode: str = "off"


# Canonical local feature extractor — FIXED and hardware-independent.
# The extractor defines the CONTENT of the local-feature database: ALIKED and
# RDD descriptors are NOT cross-matchable, so if this varied with hardware,
# databases built on different machines would stop being interchangeable.
# Override in user_config.json (models.local_extractor) ONLY if every database
# is rebuilt with the same value.
CANONICAL_LOCAL_EXTRACTOR = "aliked"


def get_default_local_extractor() -> str:
    """Return the fixed default local extractor (hardware-INDEPENDENT).

    Historically this probed VRAM and returned "aliked" on <8 GB GPUs and "rdd"
    otherwise — which made the database schema/content depend on the machine.
    Removed on purpose: the choice is now a constant so every machine builds an
    interchangeable database. See ``CANONICAL_LOCAL_EXTRACTOR``.
    """
    return CANONICAL_LOCAL_EXTRACTOR


class ModelsConfig(BaseModel):
    # Явний режим пристрою — керується конфігом, не кодом:
    #   "auto" — CUDA якщо доступна, інакше CPU-фолбек (локалізація повільна);
    #   "cuda" — форс GPU, чітка помилка на старті якщо CUDA недоступна;
    #   "cpu"  — форс CPU (повний робочий фолбек лише для локалізації;
    #            збудову/модифікацію БД усе одно робити на GPU-машині).
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    # Legacy-аліас: use_cuda:false = device:"cpu". Лишений для сумісності зі
    # старими user_config.json; нове — через models.device.
    use_cuda: bool = True
    local_extractor: str = Field(
        default_factory=get_default_local_extractor
    )  # "aliked" | "rdd" | "xfeat"
    yolo: YoloConfig = YoloConfig()
    depth_estimator: DepthEstimatorConfig = DepthEstimatorConfig()
    xfeat: XFeatSettings = XFeatSettings()
    aliked: AlikedSettings = AlikedSettings()
    rdd: RddSettings = RddSettings()
    superpoint: SuperPointSettings = SuperPointSettings()
    # git backend: official weights download into TORCH_HOME (models/.cache)
    lightglue: LightGlueSettings = LightGlueSettings()
    lightglue_superpoint: LightGlueSettings = LightGlueSettings()
    lightglue_rdd: LightGlueSettings = LightGlueSettings(model_path="models/RDD_lg-v2.pth")
    lightglue_sift: LightGlueSettings = LightGlueSettings()
    cesp: CespConfig = CespConfig()
    vlad: VladConfig = VladConfig()
    vram_management: VramManagementConfig = VramManagementConfig()
    performance: PerformanceConfig = PerformanceConfig()
    engines_cache: ModelsCacheConfig = ModelsCacheConfig()
