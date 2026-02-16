#!/usr/bin/env python3
"""
Main application entry point for Drone Topometric Localization System
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from src.gui.main_window import MainWindow
from src.utils.logging_utils import setup_logging, get_logger
# Initialize logging before any other imports
setup_logging(log_level="INFO", log_file="logs/app.log")
logger = get_logger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("DRONE TOPOMETRIC LOCALIZATION SYSTEM STARTING")
    logger.info("=" * 80)

    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

        app = QApplication(sys.argv)
        app.setApplicationName("Drone Localization")
        app.setOrganizationName("UAV Systems")

        logger.info("Qt Application initialized")
        logger.info(f"Application name: {app.applicationName()}")
        logger.info(f"Organization: {app.organizationName()}")

        window = MainWindow()
        window.show()

        logger.info("Main window created and shown")
        logger.success("Application startup complete")

        exit_code = app.exec()

        logger.info(f"Application exiting with code: {exit_code}")
        logger.info("=" * 80)

        sys.exit(exit_code)

    except Exception as e:
        logger.critical(f"Fatal error during application startup: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
