import asyncio
import json

import websockets
from websockets.server import WebSocketServerProtocol

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class WebSocketServer:
    """Asynchronous WebSocket server for coordinates telemetry broadcasting.

    Security: default host is 127.0.0.1 (local clients only). To allow external
    network access, specify host="0.0.0.0" and an api_token explicitly.
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

        # Telemetry protection: require authentication token when binding to
        # routable network interfaces (non-loopback host).
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

        # Token authentication: ?token=... query parameter or Authorization: Bearer ... header
        if self.api_token:
            supplied = None
            if "token=" in path:
                supplied = path.split("token=", 1)[1].split("&", 1)[0]
            headers = getattr(getattr(websocket, "request", None), "headers", {}) or {}
            auth = headers.get("Authorization", "") if hasattr(headers, "get") else ""
            if auth.startswith("Bearer "):
                supplied = auth[7:]
            if supplied != self.api_token:
                logger.warning(f"WebSocket auth failed from {websocket.remote_address} — closing")
                await websocket.close(code=4401, reason="Unauthorized")
                return

        logger.info(f"WebSocket client connected: {websocket.remote_address}")
        self.clients.add(websocket)
        try:
            async for message in websocket:
                # Currently clients are read-only listeners, but command handling can be added here
                logger.debug(f"Received message from client: {message}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info(f"WebSocket client disconnected: {websocket.remote_address}")

    def _build_ssl_context(self):
        """Creates and configures SSLContext for encrypted WSS connections."""
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
        self.server = await websockets.serve(self.handler, self.host, self.port, ssl=ssl_ctx)

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
            # Broadcast message to all connected clients
            await asyncio.gather(
                *[client.send(msg_str) for client in self.clients], return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Error broadcasting WebSocket message: {e}")
