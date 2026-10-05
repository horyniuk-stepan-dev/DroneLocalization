"""Background creation of a map-aligned reference-video mosaic."""

import hashlib
import json
from pathlib import Path

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from src.geometry.local_panorama import render_local_panorama
from src.security.project_scan import assert_project_writable
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LocalPanoramaWorker(QThread):
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, video_path, output_path, layout, source_sha256=""):
        super().__init__()
        self.video_path = str(video_path)
        self.output_path = str(output_path)
        self.layout = layout
        self.source_sha256 = source_sha256
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            path = Path(self.output_path)
            if path.suffix.lower() != ".png":
                raise ValueError("Local panoramas must be saved as PNG")
            assert_project_writable(path)
            assert_project_writable(str(path) + ".json")
            if self.source_sha256:
                self.progress.emit(0, "Перевірка відповідності відео базі…")
                digest = hashlib.sha256()
                with open(self.video_path, "rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        if not self._is_running:
                            return
                        digest.update(chunk)
                if digest.hexdigest() != self.source_sha256:
                    raise ValueError(
                        "Відео не відповідає тому, з якого побудована база цього шару."
                    )
            image = render_local_panorama(
                self.video_path,
                self.layout,
                running=lambda: self._is_running,
                progress=self.progress.emit,
            )
            if image is None or not self._is_running:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise ValueError("Cannot encode local panorama")
            path.write_bytes(encoded.tobytes())
            Path(str(path) + ".json").write_text(
                json.dumps(self.layout.metadata(), indent=2), encoding="utf-8"
            )
            self.progress.emit(100, "Панораму локальної карти збережено")
            self.completed.emit(str(path))
        except Exception as exc:
            logger.exception("Local panorama generation failed")
            self.error.emit(str(exc))
