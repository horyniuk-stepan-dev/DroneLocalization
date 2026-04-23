from aiohttp import web
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class RestApiServer:
    """Легкий HTTP-сервер для REST API координат."""
    
    def __init__(self, broker, host="0.0.0.0", port=8080):
        self.broker = broker
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner = None
        self.site = None
        
        self.app.add_routes([
            web.get('/api/position', self.get_position),
            web.get('/api/objects', self.get_objects),
            web.get('/api/trajectory', self.get_trajectory),
            web.get('/api/status', self.get_status)
        ])
        
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
