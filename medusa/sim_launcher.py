#!/usr/bin/env python3
"""
=============================================================================
  MedusaAlfa — Sim Launcher
  Піднімає WebGL-симулятор дрона (sim/server.py) одразу як частину програми
=============================================================================

Симулятор — це два сервери в одному процесі:
    * HTTP  http://127.0.0.1:8000  -> index.html, app.js, styles.css
    * WS    ws://127.0.0.1:8765    -> релей векторів [vx, vy, vz, yaw, action]

`SimLauncher` запускає sim/server.py окремим процесом (не заважає OpenCV та
asyncio-потоку DroneLink), чекає поки порти почнуть відповідати, відкриває
браузер і гарантовано зупиняє сервер при виході з програми.

Якщо сервер уже запущений (наприклад, ви відкрили його в іншому терміналі) —
лаунчер це побачить і не буде піднімати другий.
=============================================================================
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .paths import SIM_SERVER

DEFAULT_HTTP_PORT = 8000
DEFAULT_WS_PORT = 8765
DEFAULT_SIM_HOST = "127.0.0.1"


def port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    """True, якщо на host:port хтось приймає TCP-з'єднання."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def free_port(host: str = DEFAULT_SIM_HOST) -> int:
    """Вільний TCP-порт (для тестів)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@dataclass
class SimStatus:
    """Результат запуску симулятора."""

    http_url: str
    ws_url: str
    started: bool            # True -> сервер підняли ми
    already_running: bool    # True -> сервер уже працював до нас
    process: Optional[subprocess.Popen] = None

    def describe(self) -> str:
        if self.started:
            return f"симулятор запущено: {self.http_url}"
        if self.already_running:
            return f"симулятор уже працює: {self.http_url}"
        return "симулятор не запущено"


class SimLauncher:
    """Запуск/зупинка локального симулятора дрона."""

    def __init__(
        self,
        host: str = DEFAULT_SIM_HOST,
        http_port: int = DEFAULT_HTTP_PORT,
        ws_port: int = DEFAULT_WS_PORT,
        open_browser: bool = True,
        ready_timeout: float = 20.0,
        verbose: bool = True,
    ) -> None:
        self.host = host
        self.http_port = int(http_port)
        self.ws_port = int(ws_port)
        self.open_browser = open_browser
        self.ready_timeout = float(ready_timeout)
        self.verbose = verbose
        self.process: Optional[subprocess.Popen] = None
        self._opened_browser = False

    # ------------------------------------------------------------------ urls
    @property
    def http_url(self) -> str:
        return f"http://{self.host}:{self.http_port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.ws_port}"

    # ------------------------------------------------------------------ start
    def ensure_running(self) -> SimStatus:
        """Повертає статус симулятора, за потреби піднімає його."""
        if port_open(self.host, self.ws_port):
            status = SimStatus(self.http_url, self.ws_url, started=False, already_running=True)
            if self.verbose:
                print(f"🛰  {status.describe()}")
            self._open_browser_once()
            return status

        self._check_files()
        command = [
            sys.executable, "-u", str(SIM_SERVER),
            "--host", self.host,
            "--http-port", str(self.http_port),
            "--ws-port", str(self.ws_port),
        ]
        if self.verbose:
            print(f"🛰  Запускаю симулятор: {' '.join(command)}")
        try:
            self.process = subprocess.Popen(command, cwd=str(SIM_SERVER.parent))
        except OSError as exc:
            raise RuntimeError(f"Не вдалося запустити симулятор: {exc}") from exc

        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Симулятор завершився з кодом {self.process.returncode}. "
                    f"Перевірте, чи не зайняті порти {self.http_port}/{self.ws_port}, "
                    "та чи встановлено пакет 'websockets'."
                )
            if port_open(self.host, self.ws_port) and port_open(self.host, self.http_port):
                break
            time.sleep(0.15)
        else:
            raise RuntimeError(
                f"Симулятор не піднявся за {self.ready_timeout:.0f} с "
                f"(порти {self.http_port}/{self.ws_port})."
            )

        status = SimStatus(self.http_url, self.ws_url, started=True, already_running=False,
                           process=self.process)
        if self.verbose:
            print(f"🛰  {status.describe()}")
        self._open_browser_once()
        return status

    # ------------------------------------------------------------------ stop
    def stop(self) -> None:
        """Зупиняє лише той сервер, який запустили ми."""
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:  # pragma: no cover
                pass
        if self.verbose:
            print("🛰  Симулятор зупинено.")

    # ------------------------------------------------------------------ helpers
    def _open_browser_once(self) -> None:
        if not self.open_browser or self._opened_browser:
            return
        self._opened_browser = True
        try:
            webbrowser.open(self.http_url)
        except Exception:  # pragma: no cover - залежить від ОС
            pass

    @staticmethod
    def _check_files() -> None:
        if not Path(SIM_SERVER).exists():
            raise RuntimeError(f"Не знайдено файл симулятора: {SIM_SERVER}")

    def __enter__(self) -> "SimLauncher":
        self.ensure_running()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

