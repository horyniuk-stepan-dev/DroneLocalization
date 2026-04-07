#!/usr/bin/env python3
"""Drone Topometric Localization System — application entry point."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Suppress only known noisy third-party warnings, not everything
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import torch
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import QApplication

from config.config import APP_SETTINGS
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
        sys.exit(1)

    return hook


def main() -> None:
    # Logging must be initialized before anything else — including Qt
    try:
        from config.config import APP_SETTINGS
        debug_mode = APP_SETTINGS.models.performance.debug_mode
    except Exception:
        debug_mode = True # Safe default

    log_level = "INFO" if debug_mode else "CRITICAL"
    setup_logging(log_level=log_level, log_file="logs/app.log")
    logger = get_logger(__name__)

    # Route unhandled exceptions to loguru instead of silent PyQt6 crash
    sys.excepthook = _build_exception_hook(logger)

    logger.info("=" * 70)
    logger.info("DRONE TOPOMETRIC LOCALIZATION SYSTEM STARTING")
    logger.info("=" * 70)

    # System diagnostics for debugging
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

    try:
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
