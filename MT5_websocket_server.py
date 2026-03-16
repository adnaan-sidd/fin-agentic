"""APEX FOREX — WebSocket Server. Streams state to dashboard at 1Hz."""
import asyncio, json, logging
import websockets

log = logging.getLogger("WebSocket")

class WebSocketServer:
    def __init__(self, state):
        self.state = state
        self.clients = set()

    async def handler(self, ws, path=None):
        self.clients.add(ws)
        log.info(f"Dashboard connected ({len(self.clients)} clients)")
        try:
            await ws.send(json.dumps(self.state.get_dashboard_payload(), default=str))
            async for _ in ws: pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)

    async def broadcast(self, payload: dict):
        if not self.clients: return
        msg  = json.dumps(payload, default=str)
        dead = set()
        for ws in self.clients.copy():
            try:   await ws.send(msg)
            except: dead.add(ws)
        for ws in dead: self.clients.discard(ws)

    async def start(self):
        import os
        port = int(os.getenv("WEBSOCKET_PORT", "8765"))
        log.info(f"WebSocket server on ws://0.0.0.0:{port}")
        asyncio.create_task(websockets.serve(self.handler, "0.0.0.0", port))
