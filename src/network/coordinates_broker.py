import asyncio
import secrets
import threading
import time
from collections import deque

from PyQt6.QtCore import QObject, pyqtSlot

from config import NetworkApiConfig
from src.network.rest_server import RestApiServer
from src.network.ws_server import WebSocketServer
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

        # HARDENING P1-9/10: monotonic timestamp of the last successful fix,
        # drives the operating-state machine and stall detector. None = no fix
        # since the current tracking session began.
        self._last_fix_mono: float | None = None
        # HARDENING §4a: monotonic timestamp of the last *fresh keyframe anchor*
        # (a real re-localization, not an optical-flow-propagated fix). Drives
        # the anchor-staleness DEGRADED branch. None = no anchor yet this session.
        self._last_anchor_mono: float | None = None

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

        token = getattr(self.config, "api_token", "") or None

        # HARDENING P0-4: if any server would bind a routable host without a
        # token, self-heal to secure rather than crash — generate one and log
        # it so the operator can hand it to clients. Localhost stays tokenless.
        _local = ("127.0.0.1", "localhost", "::1")
        _remote_ws = self.config.ws_enabled and self.config.ws_host not in _local
        _remote_rest = self.config.rest_enabled and self.config.rest_host not in _local
        if token is None and (_remote_ws or _remote_rest):
            token = secrets.token_urlsafe(32)
            logger.warning(
                "Telemetry bound to a routable host without api_token — "
                "generated one automatically. Clients must authenticate with "
                "this token (Authorization: Bearer <token> or ?token=<token>):\n"
                f"    api_token = {token}\n"
                "Set network.api_token in user_config.json to pin a fixed token."
            )

        # HARDENING P1-7: resolve optional TLS. Fail closed — if TLS is enabled
        # but the cert/key pair is missing, do NOT silently serve plaintext.
        certfile = keyfile = None
        if getattr(self.config, "tls_enabled", False):
            certfile = getattr(self.config, "tls_certfile", "") or ""
            keyfile = getattr(self.config, "tls_keyfile", "") or ""
            if not (certfile and keyfile):
                raise ValueError(
                    "network_api.tls_enabled=True but tls_certfile/tls_keyfile "
                    "are not both set — refusing to start telemetry in plaintext."
                )
            logger.info("Telemetry TLS enabled (wss/https).")

        tasks = []
        if self.config.ws_enabled:
            self._ws_server = WebSocketServer(
                host=self.config.ws_host,
                port=self.config.ws_port,
                api_token=token,
                certfile=certfile,
                keyfile=keyfile,
            )
            tasks.append(self._ws_server.start())

        if self.config.rest_enabled:
            self._rest_server = RestApiServer(
                broker=self,
                host=self.config.rest_host,
                port=self.config.rest_port,
                api_token=token,
                certfile=certfile,
                keyfile=keyfile,
            )
            tasks.append(self._rest_server.start())

        if tasks:
            self._loop.run_until_complete(asyncio.gather(*tasks))
            # HARDENING P1-10: liveness heartbeat over WS, flag-gated.
            if getattr(self.config, "expose_operating_state", False):
                self._loop.create_task(self._heartbeat_loop())
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
            # New session starts in ACQUIRING until the first fix arrives.
            self._last_fix_mono = None
            self._last_anchor_mono = None

    def get_uptime(self) -> float:
        if self.is_tracking_active:
            return time.time() - self._tracking_start_time
        return 0.0

    def get_operating_state(self) -> dict:
        """HARDENING P1-9/10: honest operating state + stall info.

        IDLE       — tracking not active.
        ACQUIRING  — tracking active, no fix yet this session.
        LOST       — tracking active, last fix older than fix_stale_sec (stall).
        DEGRADED   — recent fix but below configured inlier/confidence floor.
        TRACKING   — recent, healthy fix.
        """
        stale_sec = getattr(self.config, "fix_stale_sec", 3.0)
        min_inl = getattr(self.config, "degraded_min_inliers", 0)
        min_conf = getattr(self.config, "degraded_min_confidence", 0.0)
        prop_stale = getattr(self.config, "propagation_stale_sec", 0.0)

        now = time.monotonic()
        age = None
        anchor_age = None if self._last_anchor_mono is None else now - self._last_anchor_mono

        if not self.is_tracking_active:
            state = "IDLE"
        elif self._last_fix_mono is None:
            state = "ACQUIRING"
        else:
            age = now - self._last_fix_mono
            if age > stale_sec:
                state = "LOST"
            else:
                last = self._last_position or {}
                inl = last.get("inliers", 0)
                conf = last.get("confidence", 1.0)
                if (min_inl and inl < min_inl) or (min_conf and conf < min_conf):
                    state = "DEGRADED"
                elif prop_stale and (self._last_anchor_mono is None or anchor_age > prop_stale):
                    # HARDENING §4a: tracking is coasting on optical-flow
                    # propagation with no fresh keyframe anchor for too long —
                    # honest DEGRADED even though the (propagated) fix clock is
                    # still fresh. Closes the content-blind gap.
                    state = "DEGRADED"
                else:
                    state = "TRACKING"

        return {
            "op_state": state,
            "last_fix_age_sec": round(age, 3) if age is not None else None,
            "stale_after_sec": stale_sec,
            "anchor_age_sec": round(anchor_age, 3) if anchor_age is not None else None,
        }

    async def _heartbeat_loop(self):
        """HARDENING P1-10: periodic liveness beacon so consumers detect a hung
        pipeline even when no position is being produced (i.e. LOST)."""
        interval = getattr(self.config, "heartbeat_interval_sec", 1.0)
        try:
            while True:
                await asyncio.sleep(interval)
                if self._ws_server is None:
                    continue
                msg = {"type": "heartbeat", "timestamp": time.time()}
                msg.update(self.get_operating_state())
                await self._ws_server.broadcast(msg)
        except asyncio.CancelledError:
            pass

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
        self._last_fix_mono = time.monotonic()  # HARDENING P1-9/10: stall clock
        self._history.append(msg)
        self._broadcast(msg)

    @pyqtSlot()
    def on_anchor_fix(self):
        """HARDENING §4a: a fresh keyframe anchor landed (a real re-localization,
        not an optical-flow-propagated fix). Refreshes the anchor-staleness clock
        the DEGRADED branch watches. Wired to the worker's ``anchor_fix`` signal."""
        self._last_anchor_mono = time.monotonic()

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
                    "conf": o.confidence,
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
