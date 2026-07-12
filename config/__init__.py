"""Unified application configuration (Pydantic), split into domain modules.

Import names directly from ``config`` (e.g. ``from config import get_cfg,
APP_SETTINGS``). The old ``config.config`` module remains as a deprecated
shim for one release.
"""

from config.access import (
    APP_CONFIG,
    APP_SETTINGS,
    CONFIG_FILE_PATH,
    get_active_descriptor_cfg,
    get_cfg,
    load_user_config,
    save_user_config,
)
from config.app import (
    AppConfig,
    DebugViewsConfig,
    GuiConfig,
    LiveStreamConfig,
    NetworkApiConfig,
    ObjectTrackingConfig,
    PreprocessingConfig,
)
from config.database import DatabaseConfig
from config.graph import GraphOptimizationConfig, ProjectionConfig, PropagationConfig
from config.localization import (
    ConfidenceConfig,
    HomographyConfig,
    LocalizationConfig,
    TrackingConfig,
)
from config.models import (
    CespConfig,
    Dinov2ModelConfig,
    Dinov3ModelConfig,
    GlobalDescriptorConfig,
    ModelsCacheConfig,
    ModelsConfig,
    ModelSettings,
    PerformanceConfig,
    VramManagementConfig,
    YoloConfig,
    get_default_local_extractor,
)

__all__ = [
    # models
    "Dinov2ModelConfig",
    "Dinov3ModelConfig",
    "GlobalDescriptorConfig",
    "YoloConfig",
    "ModelSettings",
    "CespConfig",
    "VramManagementConfig",
    "ModelsCacheConfig",
    "PerformanceConfig",
    "ModelsConfig",
    "get_default_local_extractor",
    # database
    "DatabaseConfig",
    # localization
    "ConfidenceConfig",
    "LocalizationConfig",
    "TrackingConfig",
    "HomographyConfig",
    # graph
    "ProjectionConfig",
    "GraphOptimizationConfig",
    "PropagationConfig",
    # app
    "PreprocessingConfig",
    "GuiConfig",
    "DebugViewsConfig",
    "ObjectTrackingConfig",
    "LiveStreamConfig",
    "NetworkApiConfig",
    "AppConfig",
    # access
    "get_cfg",
    "get_active_descriptor_cfg",
    "load_user_config",
    "save_user_config",
    "CONFIG_FILE_PATH",
    "APP_SETTINGS",
    "APP_CONFIG",
]
