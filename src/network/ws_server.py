import asyncio
import json
import websockets
from websockets.server import WebSocketServerProtocol
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class WebSocketServer:
    """Асинхронний WebSocket-сервер для розсилки координат."""
    
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.clients: set[WebSocketServerProtocol] = set()
        self.server = None
        
    async def handler(self, websocket: WebSocketServerProtocol):
        path = getattr(websocket.request, "path", "") if hasattr(websocket, "request") else ""
        if path and path != "/ws/coords":
            await websocket.close()
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
            
    async def start(self):
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}...")
        self.server = await websockets.serve(self.handler, self.host, self.port)
        
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
