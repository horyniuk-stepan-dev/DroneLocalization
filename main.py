#!/usr/bin/env python3
"""
Main application entry point for Drone Topometric Localization System
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow


def main():
    """Main application entry point"""
    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("Drone Localization")
    app.setOrganizationName("UAV Systems")

    window = QMainWindow()
    window.setWindowTitle("Drone Localization - Test")
    window.resize(800, 600)
    window.show()

    print("Main window initialization not yet implemented")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
