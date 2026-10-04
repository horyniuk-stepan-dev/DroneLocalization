"""Application-level configuration and the top-level AppConfig aggregator."""

from typing import Literal

from pydantic import BaseModel

from config.database import DatabaseConfig
from config.graph import GraphOptimizationConfig, ProjectionConfig, PropagationConfig
from config.localization import HomographyConfig, LocalizationConfig, TrackingConfig
from config.models import GlobalDescriptorConfig, ModelsConfig


class PreprocessingConfig(BaseModel):
    clahe_clip_limit: float = 3.0
    clahe_tile_grid: list[int] = [8, 8]
    # histogram_matching / reference_image_path ВИДАЛЕНО (2026-10): histogram
    # matching ніколи не був реалізований в ImagePreprocessor (лише CLAHE).
    masking_strategy: str = "yolo"


# Секцію gui (video_fps, verify_display_mode, verify_label_mode) ВИДАЛЕНО
# (2026-10): діалог налаштувань показував ці поля, але жоден код їх не читав
# (FPS береться з відео, вигляд перевірки якорів фіксований).


class DebugViewsConfig(BaseModel):
    """Вікна «очима моделей» у режимі локалізації (YOLO / Depth / DINO / Матчі).

    Усе off за замовчуванням — нульовий overhead, поки жодне вікно не відкрите.
    Секція round-trip-иться у user_config.json, тож show_* зберігають стан
    видимості вікон між запусками.
    """

    max_width: int = 640  # ширина зображень у вікнах (downscale перед emit)
    depth_every_n_keyframes: int = (
        1  # частота depth-інференсу (1 = кожен keyframe; окремий GPU-прохід)
    )
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
    # COCO ids to track. YOLO detects only the dynamic classes it masks
    # (person, bicycle, car, motorcycle, bus, truck), so ids outside that set
    # never appear. Empty list = track every detected class.
    tracked_classes: list[int] = [
        0,
        1,
        2,
        3,
        5,
        7,
    ]  # COCO: person, bicycle, car, motorcycle, bus, truck
    show_on_video: bool = True  # boxes on the video widget
    show_on_map: bool = True  # markers on the map (export keeps every object)
    project_to_gps: bool = True  # False = track in the image only, no GPS


class LiveStreamConfig(BaseModel):
    """Live / file video input (src/video/video_source.py).

    The source itself comes from the project (video sources) or --source;
    enabled / source_type / rtsp_url / usb_device were never read and were
    removed (2026-10). The type is detected from the source string.
    """

    reconnect_attempts: int = 5
    reconnect_delay_sec: float = 2.0
    buffer_size: int = 1


class FlightDataConfig(BaseModel):
    """Optional drone telemetry (src/flight_data). Default: none — vision only.

    Used as a PRIOR only: heading → rotation prior of each keyframe (a wrong
    heading costs one retrieval pass, then the vision scan takes over).
    """

    source: Literal["none", "csv"] = "none"
    csv_path: str = ""
    # "generic": time_s, heading_deg, alt_agl_m, alt_msl_m, pitch_deg, roll_deg,
    # lat, lon. "flightsim": FlightSimulator telemetry.csv (timestamp, yaw_rad, alt_z).
    csv_preset: Literal["generic", "flightsim"] = "generic"
    time_offset_s: float = 0.0  # log time = video time + offset
    max_gap_s: float = 1.0  # no row within this → no sample (stale data is not used)
    use_heading: bool = True
    # Heading the TOP of the reference frames pointed to (heading-hold layers: 0).
    reference_heading_deg: float = 0.0
    # Camera mounted rotated relative to the nose (clockwise, degrees);
    # scripts/fit_yaw_offset.py estimates it from a continuous-mode replay.
    camera_yaw_offset_deg: float = 0.0


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

    # --- HARDENING P1-7: опційний TLS для WS/REST телеметрії ---
    # tls_enabled=False (дефолт) = поточна поведінка (plaintext ws://, http://).
    # Увімкнення вимагає валідних certfile+keyfile — інакше сервер НЕ стартує
    # (fail closed, як і auth). Для закритих мереж підходить self-signed.
    tls_enabled: bool = False
    tls_certfile: str = ""
    tls_keyfile: str = ""

    # --- HARDENING P1-9/10: машина операційного стану + детектор зависання ---
    # expose_operating_state=False (дефолт) = поточний вивід /api/status.
    # Увімкнено: /api/status додає op_state (IDLE/ACQUIRING/TRACKING/DEGRADED/
    # LOST) + вік останнього фіксу, а WS отримує періодичний heartbeat — щоб
    # споживач довіряв чесному сигналу "LOST", а вартовий бачив завислий процес.
    expose_operating_state: bool = False
    # Фікс, старіший за це (сек) під час активного трекінгу => LOST (зависання).
    fix_stale_sec: float = 3.0
    # Період WS-heartbeat (сек), коли expose_operating_state=True.
    heartbeat_interval_sec: float = 1.0
    # Пороги DEGRADED; 0 вимикає перевірку (дефолт: стан лише за часом фіксу).
    degraded_min_inliers: int = 0
    degraded_min_confidence: float = 0.0
    # §4a: якщо трекінг коастить на optical-flow-пропагації без свіжого
    # keyframe-якоря довше за це (сек) => DEGRADED, навіть коли годинник фіксу
    # свіжий (пропаговані фікси несуть застарілі inliers). 0 = вимкнено (дефолт).
    # Задавати з запасом над спостережуваним інтервалом свіжого якоря.
    propagation_stale_sec: float = 0.0


class AppConfig(BaseModel):
    live_stream: LiveStreamConfig = LiveStreamConfig()
    network_api: NetworkApiConfig = NetworkApiConfig()
    object_tracking: ObjectTrackingConfig = ObjectTrackingConfig()
    global_descriptor: GlobalDescriptorConfig = GlobalDescriptorConfig()
    database: DatabaseConfig = DatabaseConfig()
    localization: LocalizationConfig = LocalizationConfig()
    tracking: TrackingConfig = TrackingConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    debug_views: DebugViewsConfig = DebugViewsConfig()
    models: ModelsConfig = ModelsConfig()
    projection: ProjectionConfig = ProjectionConfig()
    homography: HomographyConfig = HomographyConfig()
    graph_optimization: GraphOptimizationConfig = GraphOptimizationConfig()
    propagation: PropagationConfig = PropagationConfig()
    flight_data: FlightDataConfig = FlightDataConfig()
