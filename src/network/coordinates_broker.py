import asyncio
import threading
import time
from collections import deque
from PyQt6.QtCore import QObject, pyqtSlot

from config.config import NetworkApiConfig
from src.network.ws_server import WebSocketServer
from src.network.rest_server import RestApiServer
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CoordinatesBroker(QObject):
    """Централізований брокер координат для всіх споживачів."""
    
    def __init__(self, config: NetworkApiConfig):
        super().__init__()
        self.config = config
        
        self._last_position: dict | None = None
        self._last_objects: list[dict] = []
        self._history: deque = deque(maxlen=1000)
        
        self._tracking_start_time: float = 0.0
        self.is_tracking_active: bool = False
        
        self._ws_server = None
        self._rest_server = None
        
        self._loop = None
        self._loop_thread = None
        
        if self.config.enabled:
            self._start_network_services()
            
    def _start_network_services(self):
        """Запускає asyncio event loop у фоновому потоці для WS/REST."""
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()
        
    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        
        tasks = []
        if self.config.ws_enabled:
            self._ws_server = WebSocketServer(host=self.config.ws_host, port=self.config.ws_port)
            tasks.append(self._ws_server.start())
            
        if self.config.rest_enabled:
            self._rest_server = RestApiServer(broker=self, host=self.config.rest_host, port=self.config.rest_port)
            tasks.append(self._rest_server.start())
            
        if tasks:
            self._loop.run_until_complete(asyncio.gather(*tasks))
            # Запускаємо безкінечний цикл для обробки підключень
            self._loop.run_forever()
            
    def stop(self):
        self.is_tracking_active = False
        if self._loop and self._loop.is_running():
            # Запускаємо зупинку серверів асинхронно
            asyncio.run_coroutine_threadsafe(self._stop_servers(), self._loop)
            # Чекаємо трохи і зупиняємо loop
            time.sleep(0.5)
            self._loop.call_soon_threadsafe(self._loop.stop)
            
    async def _stop_servers(self):
        if self._ws_server:
            await self._ws_server.stop()
        if self._rest_server:
            await self._rest_server.stop()

    def set_tracking_active(self, active: bool):
        self.is_tracking_active = active
        if active:
            self._tracking_start_time = time.time()
            
    def get_uptime(self) -> float:
        if self.is_tracking_active:
            return time.time() - self._tracking_start_time
        return 0.0

    def get_last_position(self) -> dict | None:
        return self._last_position
        
    def get_last_objects(self) -> list[dict]:
        return self._last_objects
        
    def get_history(self, limit: int = 100) -> list[dict]:
        history_list = list(self._history)
        return history_list[-limit:]

    # Слоти для підключення до RealtimeTrackingWorker

    @pyqtSlot(float, float, float, int)
    def on_location_found(self, lat: float, lon: float, confidence: float, inliers: int):
        msg = {
            "type": "position",
            "lat": lat, 
            "lon": lon,
            "confidence": confidence,
            "inliers": inliers,
            "timestamp": time.time(),
        }
        self._last_position = msg
        self._history.append(msg)
        self._broadcast(msg)
    
    @pyqtSlot(object)
    def on_objects_gps_updated(self, objects_gps: list):
        msg = {
            "type": "objects",
            "objects": [
                {
                    "track_id": o.track_id, 
                    "class": o.class_name,
                    "lat": o.lat, 
                    "lon": o.lon, 
                    "conf": o.confidence
                }
                for o in objects_gps
            ],
            "timestamp": time.time(),
        }
        self._last_objects = msg["objects"]
        self._broadcast(msg)
    
    def _broadcast(self, msg: dict):
        if self._ws_server and self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._ws_server.broadcast(msg), self._loop)
