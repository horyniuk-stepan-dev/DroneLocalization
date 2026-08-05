import threading
import time
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VideoSourceType(Enum):
    FILE = "file"  # /path/to/video.mp4
    RTSP = "rtsp"  # rtsp://ip:port/stream
    RTMP = "rtmp"  # rtmp://ip/live/stream
    USB = "usb"  # device index (0, 1, ...)
    HTTP = "http"  # http://ip/mjpeg


@dataclass
class VideoSourceConfig:
    source: str
    source_type: VideoSourceType = VideoSourceType.FILE
    reconnect_attempts: int = 5
    reconnect_delay_sec: float = 2.0
    buffer_size: int = 1  # For live sources: 1-frame buffer (minimum latency)
    read_timeout_sec: float = 10.0
    # For live: background reader keeps only the most recent frame (drop-late).
    # Without it a consumer slower than the stream chronically lags behind real time.
    drop_late_frames: bool = True


class VideoSource:
    """Thin wrapper over cv2.VideoCapture with auto-reconnect and source-type detection."""

    def __init__(self, config: VideoSourceConfig):
        self.config = config
        self._cap = None
        self._fps = 30.0
        self._is_open = False

        # Background drop-late reader (live sources only)
        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_seq = 0
        self._consumed_seq = 0

        # Auto-detect source type if not specified explicitly
        if self.config.source_type == VideoSourceType.FILE:
            source_lower = str(self.config.source).lower()
            if source_lower.startswith("rtsp://"):
                self.config.source_type = VideoSourceType.RTSP
            elif source_lower.startswith("rtmp://"):
                self.config.source_type = VideoSourceType.RTMP
            elif source_lower.startswith("http://") or source_lower.startswith("https://"):
                self.config.source_type = VideoSourceType.HTTP
            elif source_lower.startswith("usb:") or source_lower.isdigit():
                self.config.source_type = VideoSourceType.USB
                # Strip the prefix
                if source_lower.startswith("usb:"):
                    self.config.source = self.config.source[4:]

        self._connect()

        if self._is_open and self.is_live and self.config.drop_late_frames:
            self._start_reader()

    def _start_reader(self):
        """Starts a background thread that continuously pulls frames and retains only the latest."""
        if self._reader_thread is not None:
            return
        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="VideoSourceReader", daemon=True
        )
        self._reader_thread.start()
        logger.info("Drop-late reader thread started for live source.")

    def _reader_loop(self):
        """Reads the stream as fast as possible; stores only the most recent frame."""
        failures = 0
        while not self._stop_event.is_set():
            cap = self._cap
            if cap is None:
                break

            ret, frame = cap.read()

            if ret:
                failures = 0
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_seq += 1
                continue

            if self._stop_event.is_set():
                break

            # Counter tracks CONSECUTIVE failed reads, not failed open() calls:
            # an RTSP server can accept connections without delivering frames.
            failures += 1
            if failures > self.config.reconnect_attempts:
                logger.error("Failed to reconnect after multiple attempts.")
                self._is_open = False
                return

            logger.warning(
                f"Live stream read failed. Reconnect attempt "
                f"{failures}/{self.config.reconnect_attempts}..."
            )
            if self._stop_event.wait(self.config.reconnect_delay_sec):
                return
            self._connect()

    def _read_latest(self) -> tuple[bool, np.ndarray | None]:
        """Returns the most recent frame from the background reader; waits for a new one, never repeats the old one."""
        deadline = time.monotonic() + self.config.read_timeout_sec
        while True:
            with self._frame_lock:
                if self._latest_seq > self._consumed_seq:
                    self._consumed_seq = self._latest_seq
                    return True, self._latest_frame

            if not self._is_open or self._stop_event.is_set():
                return False, None

            if time.monotonic() >= deadline:
                logger.error(
                    f"No frame from live source within {self.config.read_timeout_sec:.1f}s."
                )
                return False, None

            time.sleep(0.002)

    def _connect(self):
        """Connects to the source. Configures buffer size for live sources."""
        if self._cap is not None:
            self._cap.release()

        source_val = (
            int(self.config.source)
            if self.config.source_type == VideoSourceType.USB
            else self.config.source
        )

        logger.info(
            f"Connecting to video source: {source_val} (Type: {self.config.source_type.name})"
        )

        self._cap = cv2.VideoCapture(source_val)

        if not self._cap.isOpened():
            self._is_open = False
            logger.error(f"Failed to open video source: {source_val}")
            return

        self._is_open = True

        # Minimise buffering for live streams
        if self.is_live:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

        # Read FPS
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps > 0 and fps < 120:
            self._fps = fps
        else:
            self._fps = 30.0  # Fallback

        logger.info(f"Successfully connected to video source. FPS: {self._fps:.2f}")

    @property
    def is_live(self) -> bool:
        """True for RTSP/RTMP/USB/HTTP (no end-of-stream, no sync-sleep)."""
        return self.config.source_type in [
            VideoSourceType.RTSP,
            VideoSourceType.RTMP,
            VideoSourceType.USB,
            VideoSourceType.HTTP,
        ]

    @property
    def fps(self) -> float:
        """Stream FPS (from metadata for live sources, from header for files)."""
        return self._fps

    @property
    def is_opened(self) -> bool:
        return self._is_open

    @property
    def pos_msec(self) -> float:
        """Current video position in ms (0.0 if codec does not report / closed)."""
        if self._cap is None:
            return 0.0
        return float(self._cap.get(cv2.CAP_PROP_POS_MSEC))

    @property
    def pos_frames(self) -> float:
        """Current frame number (0.0 if closed)."""
        if self._cap is None:
            return 0.0
        return float(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Reads a frame with auto-reconnect on connection loss."""
        if not self._is_open:
            return False, None

        if self._reader_thread is not None:
            return self._read_latest()

        ret, frame = self._cap.read()

        if not ret and self.is_live:
            # For live streams: try to reconnect
            logger.warning("Connection lost to live stream. Attempting to reconnect...")
            for attempt in range(self.config.reconnect_attempts):
                time.sleep(self.config.reconnect_delay_sec)
                logger.info(f"Reconnect attempt {attempt + 1}/{self.config.reconnect_attempts}...")
                self._connect()
                if self._is_open:
                    ret, frame = self._cap.read()
                    if ret:
                        logger.success("Reconnected successfully.")
                        return True, frame

            logger.error("Failed to reconnect after multiple attempts.")
            self._is_open = False
            return False, None

        return ret, frame

    def release(self):
        """Release all resources."""
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False
