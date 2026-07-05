from aiohttp import web

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class RestApiServer:
    """Легкий HTTP-сервер для REST API координат.

    Безпека: дефолт — 127.0.0.1. Для доступу з мережі задайте host="0.0.0.0"
    явно та api_token (перевіряється заголовок Authorization: Bearer <token>).
    """

    def __init__(self, broker, host="127.0.0.1", port=8080, api_token: str | None = None):
        self.broker = broker
        self.host = host
        self.port = port
        self.api_token = api_token
        self.app = web.Application(middlewares=[self._auth_middleware])
        self.runner = None
        self.site = None

        if host not in ("127.0.0.1", "localhost", "::1") and not api_token:
            logger.warning(
                f"REST API binds to '{host}' WITHOUT api_token — "
                f"position/trajectory endpoints will be public on the network. "
                f"Set network.api_token or use 127.0.0.1."
            )

        self.app.add_routes([
            web.get('/api/position', self.get_position),
            web.get('/api/objects', self.get_objects),
            web.get('/api/trajectory', self.get_trajectory),
            web.get('/api/status', self.get_status)
        ])

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
            limit = int(request.query.get('limit', '100'))
        except ValueError:
            limit = 100

        history = self.broker.get_history(limit)
        return web.json_response(history)

    async def get_status(self, request):
        return web.json_response({
            "state": "tracking" if self.broker.is_tracking_active else "idle",
            "uptime_sec": self.broker.get_uptime()
        })

    async def start(self):
        logger.info(f"Starting REST API server on {self.host}:{self.port}...")
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

    async def stop(self):
        if self.runner:
            logger.info("Stopping REST API server...")
            await self.runner.cleanup()
