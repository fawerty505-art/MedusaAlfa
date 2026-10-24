#!/usr/bin/env python3
"""Low-latency client for the ESP32 TCP stream (big-endian length + JPEG).

Protocol: each frame is a 4-byte unsigned big-endian JPEG length, followed by
exactly that many JPEG bytes. The receiver thread keeps only the newest frame.
"""

import argparse
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np

DEFAULT_IP = "10.21.243.252"
DEFAULT_PORT = 5000
MAX_FRAME_BYTES = 8 * 1024 * 1024


def recv_exact(sock, length, running=lambda: True):
    """Read exactly length bytes; return None if the peer closes the socket."""
    data = bytearray()
    while len(data) < length and running():
        try:
            chunk = sock.recv(length - len(data))
        except socket.timeout:
            continue
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data) if len(data) == length else None


class ESP32Stream:
    """TCP camera source with read() -> BGR frame | None and release()."""

    def __init__(self, host=DEFAULT_IP, port=DEFAULT_PORT, connect_timeout=5.0):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.connected = None
        self.started = True
        self._socket = None
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _set_connected(self, state, message):
        if state != self.connected:
            self.connected = state
            print(f"📡 ESP32-CAM: {message}")

    def _set_frame(self, frame):
        with self.lock:
            self.frame = frame
            self.frame_id += 1

    def _update(self):
        while self.started:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(self.connect_timeout)
            try:
                sock.connect((self.host, self.port))
                sock.settimeout(1.0)
            except OSError as exc:
                sock.close()
                self._set_connected(False, f"не вдалося підключитися до {self.host}:{self.port} ({exc}), повтор...")
                time.sleep(1.0)
                continue

            self._socket = sock
            self._set_connected(True, f"підключено до {self.host}:{self.port}")
            try:
                while self.started:
                    header = recv_exact(sock, 4, lambda: self.started)
                    if header is None:
                        break
                    frame_len = struct.unpack(">I", header)[0]
                    if frame_len < 4 or frame_len > MAX_FRAME_BYTES:
                        raise ValueError(f"invalid JPEG frame length: {frame_len}")
                    jpeg = recv_exact(sock, frame_len, lambda: self.started)
                    if jpeg is None:
                        break
                    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        self._set_frame(frame)
            except (OSError, ValueError) as exc:
                if self.started:
                    print(f"⚠️ ESP32-CAM stream error: {exc}")
            finally:
                try:
                    sock.close()
                except OSError:
                    pass
                if self._socket is sock:
                    self._socket = None

            if self.started:
                self._set_connected(False, "потік перервано, перепідключення...")
                time.sleep(0.5)

    def wait_for_first_frame(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.frame is not None:
                    return True
            time.sleep(0.05)
        return False

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.started = False
        sock = self._socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        if self.thread.is_alive():
            self.thread.join(timeout=1.5)


def main():
    parser = argparse.ArgumentParser(description="Пряме TCP-відео з ESP32-CAM (без панелей)")
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"IP ESP32-CAM (default: {DEFAULT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    print(f"Підключення до TCP-потоку {args.ip}:{args.port} ... (Q або Esc — вихід)")
    cam = ESP32Stream(args.ip, args.port)
    if not cam.wait_for_first_frame(timeout=10.0):
        print("❌ Не отримано кадрів. Перевірте IP, порт та формат сервера.")
        cam.release()
        return 1

    window = "ESP32-CAM Live"
    try:
        while True:
            frame = cam.read()
            if frame is not None:
                cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
