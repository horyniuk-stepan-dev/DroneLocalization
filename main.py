#!/usr/bin/env python3

import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

os.environ["YOLO_VERBOSE"] = "False"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["TRT_LOGGER_SEVERITY"] = "3"

warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, message="xFormers is not available")

import traceback

import torch
import argparse
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import QApplication

from config.config import APP_SETTINGS, APP_CONFIG
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
        log_level = "INFO" # Safe default
    setup_logging(log_level=log_level, log_file="logs/app.log")
    logger = get_logger(__name__)

    sys.excepthook = _build_exception_hook(logger)

    logger.info("=" * 70)
    logger.info("DRONE TOPOMETRIC LOCALIZATION SYSTEM STARTING")
    logger.info("=" * 70)

    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"CUDA: {torch.version.cuda} | GPU: {gpu_name} | VRAM: {vram_total:.1f} GB")
        else:
            logger.warning(
                "CUDA not available — running on CPU. Performance will be significantly reduced."
            )
    except Exception as e:
        logger.warning(f"CUDA diagnostics failed: {e}. Continuing without GPU info.")

    parser = argparse.ArgumentParser(description="Drone Topometric Localization")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--project", type=str, help="Path to project directory (required for headless)")
    parser.add_argument("--source", type=str, help="Video source URL or path (required for headless)")
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
