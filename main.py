import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- PyInstaller: fix DLL loading & redirect caches for frozen builds ---
if getattr(sys, "frozen", False):
    import ctypes
    import glob
    from pathlib import Path

    _meipass = getattr(sys, "_MEIPASS", str(Path(sys.executable).parent))
    _app_dir = str(Path(sys.executable).parent)

    # 1) Fix tensorrt DLL loading if needed
    _trt_lib = os.path.join(_meipass, "tensorrt_libs")
    if os.path.isdir(_trt_lib):
        os.add_dll_directory(_trt_lib)
        os.environ["PATH"] = _trt_lib + ";" + os.environ.get("PATH", "")

    # 2) Redirect TORCH_HOME / HF_HOME
    _cache_dir = os.path.join(_meipass, ".cache")
    if os.path.isdir(_cache_dir):
        os.environ.setdefault("TORCH_HOME", os.path.join(_cache_dir, "torch"))
        _hf_hub = os.path.join(_cache_dir, "huggingface", "hub")
        if os.path.isdir(_hf_hub):
            os.environ.setdefault("HF_HOME", os.path.join(_cache_dir, "huggingface"))
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _hf_hub)

# WORKAROUND FOR PYINSTALLER + PYTORCH + WINDOWS WinError 1114:
# Import torch BEFORE anything else to prevent DLL conflicts
import torch

import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

os.environ["YOLO_VERBOSE"] = "False"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["TRT_LOGGER_SEVERITY"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, message="xFormers is not available")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import traceback

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import QApplication

from config import APP_SETTINGS
from src.core.headless_runner import HeadlessRunner
from src.gui.main_window import MainWindow
from src.utils.logging_utils import get_logger, setup_logging


class StartupWorker(QThread):
    def __init__(self, model_manager):
        super().__init__()
        self.model_manager = model_manager

    def run(self):
        try:
            self.model_manager.prewarm()
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Startup prewarm failed: {e}. Models will load on first use.")


def _build_exception_hook(log):
    """Return sys.excepthook that logs unhandled exceptions before exit."""

    def hook(exctype, value, tb):
        log.critical(
            "Unhandled exception caught — application will exit",
            exc_info=(exctype, value, tb),
        )

        traceback.print_exception(exctype, value, tb)
        sys.exit(1)

    return hook


def main() -> None:
    try:
        log_level = APP_SETTINGS.models.performance.log_level
    except Exception:
        log_level = "INFO"  # Safe default
    setup_logging(log_level=log_level, log_file="logs/app.log")
    logger = get_logger(__name__)

    sys.excepthook = _build_exception_hook(logger)

    logger.info("=" * 70)
    logger.info("DRONE TOPOMETRIC LOCALIZATION SYSTEM STARTING")
    logger.info("=" * 70)

    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")

    # ── Hardware auto-detection & compute auto-tuning ────────────────────────
    from src.utils.hardware_profile import HardwareProfile

    hw_profile = HardwareProfile.detect()
    hw_profile.log_summary()

    # Apply PyTorch backend optimizations (TF32, cudnn.benchmark, thread counts)
    hw_profile.apply_torch_backends()

    # Auto-tune config values if enabled
    if APP_SETTINGS.models.performance.auto_tune:
        import config as _cfg_module

        overrides = hw_profile.auto_tune(_cfg_module.APP_CONFIG)
        if overrides:
            hw_profile.apply_overrides(_cfg_module.APP_CONFIG, overrides)
            hw_profile.log_overrides(overrides)
            # Reload APP_SETTINGS from the updated dict so Pydantic models reflect changes
            _cfg_module.APP_SETTINGS = _cfg_module.AppConfig(**_cfg_module.APP_CONFIG)
            # Re-bind the module-level name used throughout the app
            globals()["APP_SETTINGS"] = _cfg_module.APP_SETTINGS
        else:
            logger.info("Auto-tune: all settings already optimal or user-customized")
    else:
        logger.info("Auto-tune disabled (models.performance.auto_tune = false)")

    parser = argparse.ArgumentParser(description="Drone Topometric Localization")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument(
        "--project", type=str, help="Path to project directory (required for headless)"
    )
    parser.add_argument(
        "--source", type=str, help="Video source URL or path (required for headless)"
    )
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--rest-port", type=int, default=8080, help="REST API port")

    args = parser.parse_args()

    try:
        if args.headless:
            logger.info("Running in headless mode")
            if not args.project or not args.source:
                logger.error("--project and --source are required in headless mode")
                sys.exit(1)

            APP_SETTINGS.network_api.ws_port = args.ws_port
            APP_SETTINGS.network_api.rest_port = args.rest_port

            runner = HeadlessRunner(args.project, args.source)
            runner.run()
            exit_code = 0
        else:
            QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
            app = QApplication(sys.argv)
            app.setApplicationName("Drone Localization")
            app.setOrganizationName("UAV Systems")
            logger.info("Qt application initialized")

            window = MainWindow()
            window.show()

            # Запускаємо prewarm у фоновому потоці
            if hasattr(window, "model_manager") and window.model_manager:
                app._startup_worker = StartupWorker(window.model_manager)
                app._startup_worker.start()

            logger.success("Application startup complete")

            exit_code = app.exec()

    except Exception as e:
        logger.critical(f"Fatal error during startup: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Application exiting | code={exit_code}")
    logger.info("=" * 70)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
