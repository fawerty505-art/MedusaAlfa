#!/usr/bin/env python3
"""
=============================================================================
  MedusaAlfa — Drone Link
  Фоновий WebSocket-клієнт до симулятора дрона (ws://127.0.0.1:8765)
=============================================================================

Навіщо окремий потік з власним asyncio-циклом:
    * головний цикл OpenCV не має чекати на мережу (жодного падіння FPS);
    * команда надсилається з фіксованою частотою (за замовчуванням 20 Гц),
      навіть якщо кадр оброблявся довше — симулятор має safety-timeout 0.5 с,
      тому "свіжий" вектор потрібен постійно;
    * перепідключення відбувається автоматично, якщо симулятор стартував пізніше
      або браузер перезавантажили.

Протокол (див. sim/server.py):
    Python -> симулятор : [v_x, v_y, v_z, yaw_rate, action]  (JSON-масив)
    симулятор -> Python : {"telemetry": {"x":..,"y":..,"z":..,"heading":..,"status":..}}
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

try:  # websockets >= 10 (і новий, і legacy API)
    import websockets
except ImportError:  # pragma: no cover
    websockets = None  # type: ignore[assignment]

DEFAULT_WS_URL = "ws://127.0.0.1:8765"


class DroneLink:
    """Потокобезпечний "писатель" команд у симулятор + читач телеметрії."""

    def __init__(
        self,
        url: str = DEFAULT_WS_URL,
        rate_hz: float = 20.0,
        reconnect_delay: float = 1.0,
        open_timeout: float = 3.0,
        verbose: bool = True,
    ) -> None:
        if websockets is None:
            raise RuntimeError(
                "Не встановлено пакет 'websockets'. Виконайте: python3 -m pip install -r requirements.txt"
            )
        self.url = url
        self.rate_hz = max(1.0, float(rate_hz))
        self.reconnect_delay = max(0.2, float(reconnect_delay))
        self.open_timeout = max(0.5, float(open_timeout))
        self.verbose = verbose

        self._pending: Optional[List[float]] = None
        self._pending_lock = threading.Lock()
        self._connected = threading.Event()
        self._stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_task: Optional[asyncio.Future] = None
        self._ws: Any = None

        self._telemetry: Optional[Dict[str, Any]] = None
        self._telemetry_at: Optional[float] = None
        self._last_error: Optional[str] = None
        self.sent_count = 0
        self.connect_count = 0

    # ------------------------------------------------------------------ життя
    def start(self) -> "DroneLink":
        """Запускає фоновий потік (ідемпотентно)."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stopping.clear()
        self._thread = threading.Thread(target=self._thread_main, name="drone-link", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Зупиняє потік і закриває WebSocket (швидко, без очікування таймаутів)."""
        self._stopping.set()
        loop = self._loop
        task = self._main_task
        if loop is not None and task is not None:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:  # цикл уже закритий
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        self._main_task = None
        self._connected.clear()

    def __enter__(self) -> "DroneLink":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # ------------------------------------------------------------------ стан
    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def telemetry(self) -> Optional[Dict[str, Any]]:
        """Остання телеметрія від симулятора (або None)."""
        return dict(self._telemetry) if self._telemetry else None

    def wait_connected(self, timeout: float = 5.0) -> bool:
        """Блокує до встановлення з'єднання (зручно для тестів)."""
        return self._connected.wait(timeout)

    def status_text(self) -> str:
        if self.connected:
            return "CONNECTED"
        return "OFFLINE" if self._last_error else "CONNECTING"

    # ------------------------------------------------------------------ команди
    def send(self, command) -> List[float]:
        """
        Ставить команду в чергу на відправку (неблокуюче).

        Приймає DroneCommand, список/tuple з 5 чисел або dict {"cmd": [...]}.
        Повертає нормалізований вектор, який буде надіслано.
        """
        vector = coerce_vector(command)
        with self._pending_lock:
            self._pending = vector
        return vector

    def hover(self) -> List[float]:
        """Явна команда "тримати позицію"."""
        return self.send([0.0, 0.0, 0.0, 0.0, 0])

    @property
    def pending(self) -> Optional[List[float]]:
        with self._pending_lock:
            return list(self._pending) if self._pending else None

    # ------------------------------------------------------------------ потік
    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._main_task = loop.create_task(self._run())
        try:
            loop.run_until_complete(self._main_task)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover - defensive
            self._last_error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()
            self._loop = None

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                async with websockets.connect(
                    self.url,
                    open_timeout=self.open_timeout,
                    close_timeout=1.0,
                ) as ws:
                    self._ws = ws
                    self._connected.set()
                    self.connect_count += 1
                    self._last_error = None
                    if self.verbose:
                        print(f"🔗 Drone link: підключено до {self.url}")
                    sender = asyncio.ensure_future(self._send_loop(ws))
                    receiver = asyncio.ensure_future(self._recv_loop(ws))
                    try:
                        await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for task in (sender, receiver):
                            task.cancel()
                        await asyncio.gather(sender, receiver, return_exceptions=True)
            except asyncio.CancelledError:  # pragma: no cover
                raise
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._connected.clear()
                self._ws = None

            if self._stopping.is_set():
                break
            await asyncio.sleep(self.reconnect_delay)

    async def _send_loop(self, ws) -> None:
        interval = 1.0 / self.rate_hz
        while True:
            with self._pending_lock:
                vector = list(self._pending) if self._pending else None
            if vector is not None:
                await ws.send(json.dumps(vector))
                self.sent_count += 1
            await asyncio.sleep(interval)

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            if not isinstance(raw, str):
                continue
            try:
                data = json.loads(raw)
            except ValueError:
                continue
            if isinstance(data, dict) and isinstance(data.get("telemetry"), dict):
                self._telemetry = data["telemetry"]
                self._telemetry_at = time.time()


def coerce_vector(command) -> List[float]:
    """Приводить будь-яке подання команди до [vx, vy, vz, yaw, action]."""
    if command is None:
        return [0.0, 0.0, 0.0, 0.0, 0]

    payload = getattr(command, "to_payload", None)
    if callable(payload):
        values: Sequence = payload()
    elif isinstance(command, dict):
        values = command.get("cmd", command.get("vector", ()))
    else:
        values = command

    values = list(values)
    if len(values) != 5:
        raise ValueError(f"Очікується 5 чисел [vx, vy, vz, yaw_rate, action], отримано {values!r}")
    return [round(float(values[i]), 4) for i in range(4)] + [int(values[4])]

