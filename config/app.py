"""Application-level configuration and the top-level AppConfig aggregator."""

from pydantic import BaseModel

from config.database import DatabaseConfig
from config.graph import GraphOptimizationConfig, ProjectionConfig, PropagationConfig
from config.localization import HomographyConfig, LocalizationConfig, TrackingConfig
from config.models import GlobalDescriptorConfig, ModelsConfig


class PreprocessingConfig(BaseModel):
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: list[int] = [8, 8]
    # УВАГА: histogram matching ще НЕ реалізований в ImagePreprocessor
    # (працює лише CLAHE). Прапорець вимкнено, щоб не вводити в оману.
    histogram_matching: bool = False
    reference_image_path: str = "config/reference_style.png"
    masking_strategy: str = "yolo"


class GuiConfig(BaseModel):
    video_fps: int = 30
    verify_display_mode: str = "center"  # "center" | "center_corners" | "full"
    verify_label_mode: str = "number"


class DebugViewsConfig(BaseModel):
    """Вікна «очима моделей» у режимі локалізації (YOLO / Depth / DINO / Матчі).

    Усе off за замовчуванням — нульовий overhead, поки жодне вікно не відкрите.
    Секція round-trip-иться у user_config.json, тож show_* зберігають стан
    видимості вікон між запусками.
    """

    max_width: int = 640  # ширина зображень у вікнах (downscale перед emit)
    depth_every_n_keyframes: int = 1  # частота depth-інференсу (1 = кожен keyframe; окремий GPU-прохід)
    dino_pca_enabled: bool = True  # PCA патч-токенів (інакше — лише панель retrieval)
    # Стан видимості вікон (відновлюється при старті, зберігається при виході)
    show_yolo: bool = False
    show_depth: bool = False
    show_dino: bool = False
    show_matches: bool = False


class ObjectTrackingConfig(BaseModel):
    enabled: bool = True
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    tracked_classes: list[int] = [
        0,
        1,
        2,
        3,
        5,
        7,
    ]  # COCO: person, bicycle, car, motorcycle, bus, truck
    show_on_video: bool = True
    show_on_map: bool = True
    project_to_gps: bool = True


class LiveStreamConfig(BaseModel):
    enabled: bool = False
    source_type: str = "file"  # "file" | "rtsp" | "usb"
    rtsp_url: str = ""
    usb_device: int = 0
    reconnect_attempts: int = 5
    reconnect_delay_sec: float = 2.0
    buffer_size: int = 1


class NetworkApiConfig(BaseModel):
    enabled: bool = True
    ws_enabled: bool = True
    # БЕЗПЕКА: дефолт — лише локальні клієнти. Телеметрія дрона на 0.0.0.0
    # без токена читається будь-ким у тій самій мережі. Для зовнішнього
    # доступу задайте 0.0.0.0 явно РАЗОМ з api_token.
    ws_host: str = "127.0.0.1"
    ws_port: int = 8765
    rest_enabled: bool = True
    rest_host: str = "127.0.0.1"
    rest_port: int = 8081
    # Спільний токен для WS (?token=... або Authorization: Bearer) і REST
    # (Authorization: Bearer). Порожній = без автентифікації (тільки локально!)
    api_token: str = ""


class AppConfig(BaseModel):
    live_stream: LiveStreamConfig = LiveStreamConfig()
    network_api: NetworkApiConfig = NetworkApiConfig()
    object_tracking: ObjectTrackingConfig = ObjectTrackingConfig()
    global_descriptor: GlobalDescriptorConfig = GlobalDescriptorConfig()
    database: DatabaseConfig = DatabaseConfig()
    localization: LocalizationConfig = LocalizationConfig()
    tracking: TrackingConfig = TrackingConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    gui: GuiConfig = GuiConfig()
    debug_views: DebugViewsConfig = DebugViewsConfig()
    models: ModelsConfig = ModelsConfig()
    projection: ProjectionConfig = ProjectionConfig()
    homography: HomographyConfig = HomographyConfig()
    graph_optimization: GraphOptimizationConfig = GraphOptimizationConfig()
    propagation: PropagationConfig = PropagationConfig()
