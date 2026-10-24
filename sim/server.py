#!/usr/bin/env python3
"""
Drone Simulator server  (part of MedusaAlfa — see README.md)
============================================================

* Serves the static web files on  http://127.0.0.1:8000
* Runs a WebSocket relay on       ws://127.0.0.1:8765

Every connected client may send a JSON vector  [v_x, v_y, v_z, yaw_rate, action]
(or {"cmd": [...]} / {"vector": [...]}). The server validates it and broadcasts it
to all OTHER clients (browser <- medusa_webcam.py / test_sender.py / any autonomy
stack). Messages that are not vectors (e.g. telemetry JSON objects sent by the
browser) are relayed to the other clients unchanged, so a Python controller can
read the drone state.

Launch alone:      python3 sim/server.py
Launch with CV:    python3 medusa_webcam.py      (starts this server automatically)
Options:           python3 sim/server.py --host 127.0.0.1 --http-port 8000 --ws-port 8765
"""

import argparse
import asyncio
import functools
import http.server
import json
import math
import os
import socketserver
import sys
import threading

try:
    import websockets
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency. Install it with:  python3 -m pip install -r requirements.txt")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8000
DEFAULT_WS_PORT = 8765
WEB_ROOT = os.path.dirname(os.path.abspath(__file__))

CLIENTS = set()


# --------------------------------------------------------------------------- #
# Static HTTP server (runs in a daemon thread)
# --------------------------------------------------------------------------- #
class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler with caching disabled so edits show up on refresh."""

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript",
        ".css": "text/css",
        ".html": "text/html",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):  # keep the terminal readable
        pass


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_http_server(host=DEFAULT_HOST, port=DEFAULT_HTTP_PORT):
    handler = functools.partial(QuietHandler, directory=WEB_ROOT)
    try:
        httpd = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        sys.exit(f"Cannot bind HTTP port {port}: {exc}")
    thread = threading.Thread(target=httpd.serve_forever, name="http", daemon=True)
    thread.start()
    return httpd


# --------------------------------------------------------------------------- #
# WebSocket relay
# --------------------------------------------------------------------------- #
def parse_vector(raw):
    """Return a clean [vx, vy, vz, yaw_rate, action] list, or None if invalid."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if isinstance(data, dict):
        data = data.get("vector", data.get("cmd"))
    if not isinstance(data, (list, tuple)) or len(data) != 5:
        return None
    try:
        vx, vy, vz, yaw, action = (float(v) for v in data)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (vx, vy, vz, yaw, action)):
        return None
    return [vx, vy, vz, yaw, int(action)]


async def broadcast(message, sender):
    targets = [c for c in CLIENTS if c is not sender]
    if not targets:
        return
    await asyncio.gather(*(c.send(message) for c in targets), return_exceptions=True)


async def ws_handler(websocket, path=None):  # `path` keeps old websockets versions happy
    CLIENTS.add(websocket)
    peer = websocket.remote_address
    print(f"[ws] client connected    {peer}  (total {len(CLIENTS)})")
    try:
        async for raw in websocket:
            if isinstance(raw, bytes):
                continue
            vector = parse_vector(raw)
            if vector is not None:
                await broadcast(json.dumps(vector), websocket)
                continue
            # Not a vector: relay JSON objects (telemetry etc.), drop the rest.
            try:
                payload = json.loads(raw)
            except ValueError:
                continue
            if isinstance(payload, dict):
                await broadcast(raw, websocket)
    except websockets.ConnectionClosed:
        pass
    finally:
        CLIENTS.discard(websocket)
        print(f"[ws] client disconnected {peer}  (total {len(CLIENTS)})")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Drone simulator: static web UI + WebSocket relay")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"bind address (default: {DEFAULT_HOST})")
    parser.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT,
                        help=f"HTTP port for the web UI (default: {DEFAULT_HTTP_PORT})")
    parser.add_argument("--ws-port", type=int, default=DEFAULT_WS_PORT,
                        help=f"WebSocket relay port (default: {DEFAULT_WS_PORT})")
    parser.add_argument("--quiet", action="store_true", help="do not print the banner")
    return parser.parse_args(argv)


async def main(host=DEFAULT_HOST, http_port=DEFAULT_HTTP_PORT, ws_port=DEFAULT_WS_PORT, quiet=False):
    start_http_server(host, http_port)
    try:
        async with websockets.serve(ws_handler, host, ws_port):
            if not quiet:
                shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
                print("=" * 64)
                print("  Drone Simulator running")
                print(f"  Open in browser     : http://{shown}:{http_port}")
                print(f"  WebSocket           : ws://{shown}:{ws_port}")
                print("  Test sender         : python3 sim/test_sender.py")
                print("  With computer vision: python3 medusa_webcam.py")
                print("  Press Ctrl+C to stop")
                print("=" * 64)
            await asyncio.Future()  # run forever
    except OSError as exc:
        sys.exit(f"Cannot bind WebSocket port {ws_port}: {exc}")


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(main(args.host, args.http_port, args.ws_port, args.quiet))
    except KeyboardInterrupt:
        print("\nShutting down.")
