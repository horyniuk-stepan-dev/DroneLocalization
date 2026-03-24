# config/config.py
#
# Єдиний конфіг для всього застосунку з валідацією через Pydantic.

from typing import Any

from pydantic import BaseModel


class Dinov2Config(BaseModel):
    descriptor_dim: int = 1024
    input_size: int = 336


class DatabaseConfig(BaseModel):
    frame_step: int = 3
    prefetch_queue_size: int = 32
    keypoint_video_scale: float = 0.5
    inter_frame_min_matches: int = 15
    inter_frame_ransac_thresh: float = 3.0


class ConfidenceConfig(BaseModel):
    inlier_weight: float = 0.7
    stability_weight: float = 0.3
    rmse_norm_m: float = 10.0
    disagreement_norm_m: float = 5.0
    confidence_max_inliers: int = 80


class LocalizationConfig(BaseModel):
    min_matches: int = 12
    min_inliers_accept: int = 10
    ratio_threshold: float = 0.85
    ransac_threshold: float = 3.0
    retrieval_top_k: int = 12
    early_stop_inliers: int = 40
    retrieval_only_min_score: float = 0.90
    auto_rotation: bool = True
    enable_lightglue_fallback: bool = True
    fallback_extractor: str = "aliked"
    confidence: ConfidenceConfig = ConfidenceConfig()


class TrackingConfig(BaseModel):
    kalman_process_noise: float = 2.0
    kalman_measurement_noise: float = 5.0
    outlier_window: int = 10
    outlier_threshold_std: float = 25.0
    max_speed_mps: float = 60.0
    max_consecutive_outliers: int = 5
    process_fps: float = 1.0


class PreprocessingConfig(BaseModel):
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: list[int] = [8, 8]
    histogram_matching: bool = True
    reference_image_path: str = "config/reference_style.png"


class GuiConfig(BaseModel):
    video_fps: int = 30


class YoloConfig(BaseModel):
    model_path: str = "yolo11x-seg.pt"
    vram_required_mb: float = 1200.0
    description: str = "YOLOv11x-seg (Extra Large) for dynamic object masking"


class ModelSettings(BaseModel):
    hub_repo: str | None = ""
    hub_model: str | None = ""
    top_k: int = 2048
    vram_required_mb: float = 500.0
    model_path: str | None = ""
    max_keypoints: int = 4096
    nms_radius: int = 4
    depth_confidence: float = -1.0
    width_confidence: float = -1.0
    detection_threshold: float = 0.001


class CespConfig(BaseModel):
    enabled: bool = False
    weights_path: str | None = None
    scales: list[int] = [1, 2, 4]


class VramManagementConfig(BaseModel):
    max_vram_ratio: float = 0.8
    default_required_mb: float = 2000.0


class PerformanceConfig(BaseModel):
    propagation_max_workers: int = 4
    fp16_enabled: bool = True


class ModelsConfig(BaseModel):
    use_cuda: bool = True
    yolo: YoloConfig = YoloConfig()
    xfeat: ModelSettings = ModelSettings(
        hub_repo="verlab/accelerated_features",
        hub_model="XFeat",
        top_k=2048,
        vram_required_mb=300.0,
    )
    aliked: ModelSettings = ModelSettings(max_keypoints=4096, vram_required_mb=400.0)
    superpoint: ModelSettings = ModelSettings(
        nms_radius=4, max_keypoints=4096, vram_required_mb=500.0
    )
    lightglue: ModelSettings = ModelSettings(vram_required_mb=1000.0)
    dinov2: ModelSettings = ModelSettings(
        hub_repo="facebookresearch/dinov2", hub_model="dinov2_vitl14", vram_required_mb=1600.0
    )
    cesp: CespConfig = CespConfig()
    vram_management: VramManagementConfig = VramManagementConfig()
    performance: PerformanceConfig = PerformanceConfig()


class ProjectionConfig(BaseModel):
    default_mode: str = "WEB_MERCATOR"
    strict_projection: bool = True
    fallback_to_webmercator: bool = True
    anchor_rmse_threshold_m: float = 3.0
    anchor_max_error_m: float = 5.0
    propagation_disagreement_threshold_m: float = 2.0
    localizer_sample_points: int = 9
    localizer_expected_spread_m: float = 150.0


class AppConfig(BaseModel):
    dinov2: Dinov2Config = Dinov2Config()
    database: DatabaseConfig = DatabaseConfig()
    localization: LocalizationConfig = LocalizationConfig()
    tracking: TrackingConfig = TrackingConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    gui: GuiConfig = GuiConfig()
    models: ModelsConfig = ModelsConfig()
    projection: ProjectionConfig = ProjectionConfig()


def get_cfg(config: Any, path: str, default: Any = None) -> Any:
    """Централізований доступ до конфігу з dot-path.
    Працює як зі словниками, так і з Pydantic-моделями.
    """
    keys = path.split(".")
    current = config

    for key in keys:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            return default
    return current


# Екземпляр конфігу за замовчуванням
APP_CONFIG = AppConfig().model_dump()
# Також надаємо доступ як до об'єкта для нових модулів
APP_SETTINGS = AppConfig()
