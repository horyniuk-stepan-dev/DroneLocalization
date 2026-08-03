from aiohttp import web

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RestApiServer:
    """Легкий HTTP-сервер для REST API координат.

    Безпека: дефолт — 127.0.0.1. Для доступу з мережі задайте host="0.0.0.0"
    явно та api_token (перевіряється заголовок Authorization: Bearer <token>).
    """

    def __init__(
        self,
        broker,
        host="127.0.0.1",
        port=8080,
        api_token: str | None = None,
        certfile: str | None = None,
        keyfile: str | None = None,
    ):
        self.broker = broker
        self.host = host
        self.port = port
        self.api_token = api_token
        self.certfile = certfile
        self.keyfile = keyfile
        self.app = web.Application(middlewares=[self._auth_middleware])
        self.runner = None
        self.site = None

        # HARDENING P0-4: fail closed (see WebSocketServer for rationale).
        if host not in ("127.0.0.1", "localhost", "::1") and not api_token:
            raise ValueError(
                f"Refusing to start REST API server on routable host '{host}' "
                f"without api_token — position/trajectory endpoints would be "
                f"public on the network. Set network.api_token or bind 127.0.0.1."
            )

        self.app.add_routes(
            [
                web.get("/api/position", self.get_position),
                web.get("/api/objects", self.get_objects),
                web.get("/api/trajectory", self.get_trajectory),
                web.get("/api/status", self.get_status),
            ]
        )

    @web.middleware
    async def _auth_middleware(self, request, handler):
        if self.api_token:
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {self.api_token}":
                return web.json_response({"error": "Unauthorized"}, status=401)
        return await handler(request)

    async def get_position(self, request):
        pos = self.broker.get_last_position()
        if pos:
            return web.json_response(pos)
        return web.json_response({"error": "No position data yet"}, status=404)

    async def get_objects(self, request):
        objects = self.broker.get_last_objects()
        return web.json_response(objects)

    async def get_trajectory(self, request):
        try:
            limit = int(request.query.get("limit", "100"))
        except ValueError:
            limit = 100

        history = self.broker.get_history(limit)
        return web.json_response(history)

    async def get_status(self, request):
        resp = {
            "state": "tracking" if self.broker.is_tracking_active else "idle",
            "uptime_sec": self.broker.get_uptime(),
        }
        # HARDENING P1-9/10: additive, flag-gated operating-state + stall info.
        if getattr(self.broker.config, "expose_operating_state", False):
            resp.update(self.broker.get_operating_state())
        return web.json_response(resp)

    def _build_ssl_context(self):
        """HARDENING P1-7: TLS context or None (see WebSocketServer). Fail closed."""
        if not (self.certfile and self.keyfile):
            return None
        import ssl

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=self.certfile, keyfile=self.keyfile)
        return ctx

    async def start(self):
        ssl_ctx = self._build_ssl_context()
        scheme = "https" if ssl_ctx else "http"
        logger.info(f"Starting REST API server on {scheme}://{self.host}:{self.port}...")
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port, ssl_context=ssl_ctx)
        await self.site.start()

    async def stop(self):
        if self.runner:
            logger.info("Stopping REST API server...")
            await self.runner.cleanup()
