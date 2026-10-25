"""Tests for the ESP32 4-byte big-endian length-prefixed TCP stream."""

import os
import socket
import socketserver
import struct
import sys
import threading
import time
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from medusa.esp32_stream import DEFAULT_IP, DEFAULT_PORT, ESP32Stream  # noqa: E402


def make_jpeg(value):
    image = np.full((240, 320, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


class FakeESP32Handler(socketserver.BaseRequestHandler):
    def handle(self):
        handshake = bytearray()
        while not handshake.endswith(b"\n"):
            chunk = self.request.recv(1)
            if not chunk:
                return
            handshake.extend(chunk)
        self.server.handshakes.append(bytes(handshake))
        if handshake != b"START\n":
            return

        for i in range(300):
            jpeg = make_jpeg((i * 10) % 256)
            packet = struct.pack(">I", len(jpeg)) + jpeg
            try:
                self.request.sendall(packet)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            time.sleep(0.02)


class FakeESP32Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self.handshakes = []
        super().__init__(*args, **kwargs)


class ESP32StreamTests(unittest.TestCase):
    def setUp(self):
        self.server = FakeESP32Server(("127.0.0.1", 0), FakeESP32Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.host, self.port = self.server.server_address
        self.stream = None

    def tearDown(self):
        if self.stream is not None:
            self.stream.release()
        self.server.shutdown()
        self.server.server_close()

    def test_defaults_match_supplied_connection_example(self):
        self.assertEqual(DEFAULT_IP, "10.21.246.241")
        self.assertEqual(DEFAULT_PORT, 80)

    def test_receives_first_length_prefixed_jpeg_frame(self):
        self.stream = ESP32Stream(self.host, self.port)
        self.assertTrue(self.stream.wait_for_first_frame(timeout=5.0))
        frame = self.stream.read()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (320, 240, 3))
        self.assertEqual(self.server.handshakes, [b"START\n"])

    def test_receiver_continues_updating_latest_frame(self):
        self.stream = ESP32Stream(self.host, self.port)
        self.assertTrue(self.stream.wait_for_first_frame(timeout=5.0))
        first_id = self.stream.frame_id
        time.sleep(0.3)
        self.assertGreater(self.stream.frame_id, first_id)

    def test_read_returns_copy(self):
        self.stream = ESP32Stream(self.host, self.port)
        self.assertTrue(self.stream.wait_for_first_frame(timeout=5.0))
        frame = self.stream.read()
        frame[:] = 0
        stored = self.stream.read()
        self.assertFalse(np.array_equal(stored, np.zeros_like(stored)))

    def test_unreachable_server_does_not_crash(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.stream = ESP32Stream("127.0.0.1", port, connect_timeout=0.5)
        self.assertFalse(self.stream.wait_for_first_frame(timeout=1.0))
        self.assertIsNone(self.stream.read())


if __name__ == "__main__":
    unittest.main()
