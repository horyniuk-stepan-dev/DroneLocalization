import time
import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class VideoSourceType(Enum):
    FILE = "file"           # /path/to/video.mp4
    RTSP = "rtsp"           # rtsp://ip:port/stream
    RTMP = "rtmp"           # rtmp://ip/live/stream
    USB = "usb"             # device index (0, 1, ...)
    HTTP = "http"           # http://ip/mjpeg

@dataclass
class VideoSourceConfig:
    source: str
    source_type: VideoSourceType = VideoSourceType.FILE
    reconnect_attempts: int = 5
    reconnect_delay_sec: float = 2.0
    buffer_size: int = 1        # Для live: буфер 1 кадр (мінімальна затримка)
    read_timeout_sec: float = 10.0

class VideoSource:
    """Обгортка над cv2.VideoCapture з auto-reconnect та type detection."""
    
    def __init__(self, config: VideoSourceConfig):
        self.config = config
        self._cap = None
        self._fps = 30.0
        self._is_open = False
        
        # Визначаємо тип джерела, якщо він не вказаний явно
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
                # Очищаємо префікс
                if source_lower.startswith("usb:"):
                    self.config.source = self.config.source[4:]

        self._connect()

    def _connect(self):
        """Підключається до джерела. Якщо це live, налаштовує розмір буфера."""
        if self._cap is not None:
            self._cap.release()
            
        source_val = int(self.config.source) if self.config.source_type == VideoSourceType.USB else self.config.source
        
        logger.info(f"Connecting to video source: {source_val} (Type: {self.config.source_type.name})")
        
        self._cap = cv2.VideoCapture(source_val)
        
        if not self._cap.isOpened():
            self._is_open = False
            logger.error(f"Failed to open video source: {source_val}")
            return
            
        self._is_open = True
        
        # Для live-потоків мінімізуємо буферизацію
        if self.is_live:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
            
        # Зчитуємо FPS
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps > 0 and fps < 120:
            self._fps = fps
        else:
            self._fps = 30.0  # Фолбек
            
        logger.info(f"Successfully connected to video source. FPS: {self._fps:.2f}")

    @property
    def is_live(self) -> bool:
        """True для RTSP/RTMP/USB/HTTP (немає кінця потоку, немає sync-sleep)."""
        return self.config.source_type in [
            VideoSourceType.RTSP, 
            VideoSourceType.RTMP, 
            VideoSourceType.USB, 
            VideoSourceType.HTTP
        ]
    
    @property
    def fps(self) -> float:
        """FPS потоку (для live — з метаданих, для файлу — з заголовку)."""
        return self._fps
        
    @property
    def is_opened(self) -> bool:
        return self._is_open

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Читає кадр з auto-reconnect при втраті з'єднання."""
        if not self._is_open:
            return False, None
            
        ret, frame = self._cap.read()
        
        if not ret and self.is_live:
            # Для live-потоків: пробуємо перепідключитися
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
        """Звільняє ресурси."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False
