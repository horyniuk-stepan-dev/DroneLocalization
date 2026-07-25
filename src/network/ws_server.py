import asyncio
import json

import websockets
from websockets.server import WebSocketServerProtocol

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class WebSocketServer:
    """Асинхронний WebSocket-сервер для розсилки координат.

    Безпека: дефолт — 127.0.0.1 (лише локальні клієнти). Для доступу з мережі
    задайте host="0.0.0.0" явно та api_token — телеметрія дрона не має бути
    відкритою в чужому Wi-Fi.
    """

    def __init__(
        self,
        host="127.0.0.1",
        port=8765,
        api_token: str | None = None,
        certfile: str | None = None,
        keyfile: str | None = None,
    ):
        self.host = host
        self.port = port
        self.api_token = api_token
        self.certfile = certfile
        self.keyfile = keyfile
        self.clients: set[WebSocketServerProtocol] = set()
        self.server = None

        # HARDENING P0-4: fail closed. Binding drone telemetry to a routable
        # host without a token would leave position readable by anyone on the
        # network — refuse instead of merely warning. Localhost stays
        # frictionless (no token required). Normal startup never hits this
        # because CoordinatesBroker auto-generates a token for remote hosts;
        # this guards direct/headless/test instantiation.
        if host not in ("127.0.0.1", "localhost", "::1") and not api_token:
            raise ValueError(
                f"Refusing to start WebSocket server on routable host '{host}' "
                f"without api_token — drone telemetry would be public on the "
                f"network. Set network.api_token or bind 127.0.0.1."
            )

    async def handler(self, websocket: WebSocketServerProtocol):
        path = getattr(websocket.request, "path", "") if hasattr(websocket, "request") else ""
        if path and not path.startswith("/ws/coords"):
            await websocket.close()
            return

        # Токен: ?token=... у query або заголовок Authorization: Bearer ...
        if self.api_token:
            supplied = None
            if "token=" in path:
                supplied = path.split("token=", 1)[1].split("&", 1)[0]
            headers = getattr(getattr(websocket, "request", None), "headers", {}) or {}
            auth = headers.get("Authorization", "") if hasattr(headers, "get") else ""
            if auth.startswith("Bearer "):
                supplied = auth[7:]
            if supplied != self.api_token:
                logger.warning(
                    f"WebSocket auth failed from {websocket.remote_address} — closing"
                )
                await websocket.close(code=4401, reason="Unauthorized")
                return

        logger.info(f"WebSocket client connected: {websocket.remote_address}")
        self.clients.add(websocket)
        try:
            async for message in websocket:
                # Наразі клієнти тільки слухають, але тут можна додати обробку команд
                logger.debug(f"Received message from client: {message}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info(f"WebSocket client disconnected: {websocket.remote_address}")

    def _build_ssl_context(self):
        """HARDENING P1-7: build a TLS context, or None for plaintext.

        Fail closed: if a cert/key pair is supplied it must load, otherwise we
        refuse to start rather than silently falling back to plaintext ws://.
        """
        if not (self.certfile and self.keyfile):
            return None
        import ssl

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=self.certfile, keyfile=self.keyfile)
        return ctx

    async def start(self):
        ssl_ctx = self._build_ssl_context()
        scheme = "wss" if ssl_ctx else "ws"
        logger.info(f"Starting WebSocket server on {scheme}://{self.host}:{self.port}...")
        self.server = await websockets.serve(
            self.handler, self.host, self.port, ssl=ssl_ctx
        )

    async def stop(self):
        if self.server:
            logger.info("Stopping WebSocket server...")
            self.server.close()
            await self.server.wait_closed()

    async def broadcast(self, message: dict):
        if not self.clients:
            return

        try:
            msg_str = json.dumps(message)
            # Розсилаємо повідомлення всім підключеним клієнтам
            await asyncio.gather(*[client.send(msg_str) for client in self.clients], return_exceptions=True)
        except Exception as e:
            logger.error(f"Error broadcasting WebSocket message: {e}")
