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
    model_path: str = "models/yolo11n-seg.pt"
    vram_required_mb: float = 200.0
    description: str = "YOLOv11n-seg (Nano) for dynamic object masking"


class ModelSettings(BaseModel):
    hub_repo: str | None = ""
    hub_model: str | None = ""
    top_k: int = 2048
    vram_required_mb: float = 500.0
    model_path: str | None = ""
    backend: str = "git"  # "git" | "torchscript" | "tensorrt"
    auto_convert: bool = True
    dtype: str = "float16"  # "float16" | "float32"
    max_keypoints: int = 4096
    nms_radius: int = 4
    depth_confidence: float = -1.0
    width_confidence: float = -1.0
    detection_threshold: float = 0.001


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
    auto_compile: bool = False


class PerformanceConfig(BaseModel):
    auto_tune: bool = True  # Auto-detect hardware and tune batch sizes, threads, VRAM limits
    auto_tune_vram_headroom: float = 0.0  # Extra VRAM (MB) to reserve beyond tier default (0 = auto)
    propagation_max_workers: int = 4
    fp16_enabled: bool = True
    # ADDENDUM §3 (слабкі GPU): максимальний батч ViT-форварда в
    # extract_global_descriptors_multi (recovery: до 20 кадрів разом).
    # 0 = без ліміту (ПОТОЧНА поведінка). На 4 GB VRAM безпечно 4-6.
    # Впливає лише на пік памʼяті; дескриптори побітово ті самі.
    global_batch_max: int = 0
    torch_compile: bool = False
    use_tensorrt_for_yolo: bool = False  # portable across GPUs; TRT engines are hardware-specific
    log_level: str = "INFO"
    debug_mode: bool = True


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
    use_cuda: bool = True
    local_extractor: str = Field(default_factory=get_default_local_extractor)  # "aliked" | "rdd"
    yolo: YoloConfig = YoloConfig()
    xfeat: ModelSettings = ModelSettings(
        hub_repo="verlab/accelerated_features",
        hub_model="XFeat",
        top_k=2048,
        vram_required_mb=300.0,
    )
    aliked: ModelSettings = ModelSettings(max_keypoints=4096, vram_required_mb=400.0)
    rdd: ModelSettings = ModelSettings(
        vram_required_mb=500.0,
        model_path="models/RDD-v2.pth",
        max_keypoints=4096,
    )
    superpoint: ModelSettings = ModelSettings(
        nms_radius=4, max_keypoints=4096, vram_required_mb=500.0
    )
    lightglue: ModelSettings = ModelSettings(
        vram_required_mb=800.0,
        backend="git",
        # git backend: official weights download into TORCH_HOME (models/.cache)
        model_path="",
        auto_convert=False,
    )
    lightglue_superpoint: ModelSettings = ModelSettings(
        vram_required_mb=800.0,
        backend="git",
        # git backend: official weights download into TORCH_HOME (models/.cache)
        model_path="",
        auto_convert=False,
    )
    lightglue_rdd: ModelSettings = ModelSettings(
        vram_required_mb=800.0,
        backend="git",
        model_path="models/RDD_lg-v2.pth",
        auto_convert=False,
    )
    lightglue_sift: ModelSettings = ModelSettings(
        vram_required_mb=800.0,
        backend="git",
        auto_convert=False,
    )
    cesp: CespConfig = CespConfig()
    vlad: VladConfig = VladConfig()
    vram_management: VramManagementConfig = VramManagementConfig()
    performance: PerformanceConfig = PerformanceConfig()
    engines_cache: ModelsCacheConfig = ModelsCacheConfig()
