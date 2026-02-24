import pytest
import asyncio
import json
import socket
import threading
from ai.train import start_ws_server, broadcast, METRICS_LOG

try:
    import websockets
    HAS_WS = True
except ImportError:
    HAS_WS = False


def _find_free_port():
    """Find an available port to avoid conflicts with other processes."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_dashboard_websocket_broadcast():
    if not HAS_WS:
        pytest.skip("websockets not installed")

    ws_server_port = _find_free_port()

    async def run_test():
        # Clear logs before starting server to avoid race with leftover state
        METRICS_LOG.clear()
        METRICS_LOG.append({"dummy": "history_data"})

        # Start server on dynamic port to avoid conflicts
        server_thread = threading.Thread(target=start_ws_server, args=(ws_server_port,), daemon=True)
        server_thread.start()
        
        # Give server time to bind port
        await asyncio.sleep(0.5)
        
        uri = f"ws://127.0.0.1:{ws_server_port}"
        
        # Client connects
        async with websockets.connect(uri) as websocket:
            # Upon connection, server should send metrics_history
            init_msg = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            init_data = json.loads(init_msg)
            
            assert init_data["type"] == "metrics_history"
            assert len(init_data["data"]) == 1
            assert init_data["data"][0]["dummy"] == "history_data"
            
            # Test broadcasting a live metric payload (simulating train loop)
            sample_metrics_data = {"loss": 2.5, "step": 100}
            
            # Broadcast is thread-safe using asyncio.run_coroutine_threadsafe
            broadcast("training_metrics", sample_metrics_data)
            
            # Wait for the broadcasted update
            live_msg = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            live_data = json.loads(live_msg)
            
            assert live_data["type"] == "training_metrics"
            assert live_data["loss"] == 2.5
            assert live_data["step"] == 100

    asyncio.run(run_test())
