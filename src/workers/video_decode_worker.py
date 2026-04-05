import queue
import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VideoDecodeWorker(QThread):
    """
    Фоновий потік для декодування відео та читання кадрів.
    Запобігає блокуванню головного GUI потоку під час I/O операцій.
    """

    frame_ready = pyqtSignal(int, np.ndarray)  # (frame_id, frame_bgr)
    error = pyqtSignal(str)
    video_loaded = pyqtSignal(int, float)  # (total_frames, fps)
    playback_stopped = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cmd_queue = queue.Queue()
        self._is_running = True
        self.cap = None

    def run(self):
        is_playing = False
        play_fps = 30.0
        last_play_time = 0.0

        while self._is_running:
            try:
                # Читаємо команди блокуючи чергу (з таймаутом для плейбеку)
                if is_playing:
                    # Розрахунок часу до наступного кадру
                    elapsed = time.perf_counter() - last_play_time
                    delay = max(0.001, (1.0 / play_fps) - elapsed)

                    try:
                        cmd, arg = self.cmd_queue.get(timeout=delay)
                    except queue.Empty:
                        # Час грати наступний кадр
                        cmd, arg = "next_frame", None
                else:
                    cmd, arg = self.cmd_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Обробка команди
            try:
                if cmd == "load":
                    self._internal_load(arg)
                elif cmd == "seek":
                    is_playing = False
                    self._internal_seek(arg)
                elif cmd == "play":
                    is_playing = True
                    play_fps = arg if arg > 0 else 30.0
                    last_play_time = time.perf_counter()
                elif cmd == "pause":
                    is_playing = False
                    self.playback_stopped.emit()
                elif cmd == "stop":
                    self._is_running = False
                    break
                elif cmd == "next_frame":
                    if self.cap and self.cap.isOpened():
                        ret, frame = self.cap.read()
                        if ret:
                            frame_id = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                            self.frame_ready.emit(frame_id, frame)
                            last_play_time = time.perf_counter()
                        else:
                            is_playing = False
                            self.playback_stopped.emit()

                self.cmd_queue.task_done()
            except Exception as e:
                logger.error(f"VideoDecodeWorker error handling command {cmd}: {e}")
                self.error.emit(str(e))

        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("VideoDecodeWorker thread finished.")

    def _internal_load(self, path: str):
        if self.cap:
            self.cap.release()
            self.cap = None

        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            if cap:
                cap.release()
            self.error.emit(f"Не вдалося відкрити: {path}")
            return

        self.cap = cap
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Video loaded: {total} frames, {fps} fps")
        self.video_loaded.emit(total, fps)

        # Read first frame
        self._internal_seek(0)

    def _internal_seek(self, frame_id: int):
        if not (self.cap and self.cap.isOpened()):
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()

        if not ret:
            # Fallback для деяких кодеків (шукати через час)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.cap.set(cv2.CAP_PROP_POS_MSEC, (frame_id / fps) * 1000.0)
                ret, frame = self.cap.read()

        if ret and frame is not None:
            self.frame_ready.emit(frame_id, frame)

    # --- Public API for GUI (thread-safe) ---

    def load(self, path: str):
        self.cmd_queue.put(("load", path))

    def seek(self, frame_id: int):
        # Відкидаємо попередні seek-команди, якщо їх накопичилось багато
        # Це запобігає затримкам, якщо користувач швидко тягнув повзунок
        self._clear_queue_of("seek")
        self.cmd_queue.put(("seek", frame_id))

    def play(self, fps: float):
        self.cmd_queue.put(("play", fps))

    def pause(self):
        self.cmd_queue.put(("pause", None))

    def stop(self):
        self.cmd_queue.put(("stop", None))

    def _clear_queue_of(self, cmd_to_remove: str):
        """Видаляє застарілі команди з черги (корисно для debounce)."""
        temp_list = []
        try:
            while True:
                cmd, arg = self.cmd_queue.get_nowait()
                if cmd != cmd_to_remove:
                    temp_list.append((cmd, arg))
                self.cmd_queue.task_done()
        except queue.Empty:
            pass

        for cmd, arg in temp_list:
            self.cmd_queue.put((cmd, arg))
