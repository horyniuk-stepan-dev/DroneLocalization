#!/usr/bin/env python3
"""Drone Topometric Localization System — application entry point."""
import sys
import warnings
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Suppress only known noisy third-party warnings, not everything
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning,        module="torchvision")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from src.gui.main_window import MainWindow
from src.utils.logging_utils import setup_logging, get_logger


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
    setup_logging(log_level="INFO", log_file="logs/app.log")
    logger = get_logger(__name__)

    # Route unhandled exceptions to loguru instead of silent PyQt6 crash
    sys.excepthook = _build_exception_hook(logger)

    logger.info("=" * 70)
    logger.info("DRONE TOPOMETRIC LOCALIZATION SYSTEM STARTING")
    logger.info("=" * 70)

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
