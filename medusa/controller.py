#!/usr/bin/env python3
"""
=============================================================================
  MedusaAlfa — Gesture Drone Controller
  Склеює три частини: руки (MediaPipe) -> жест (GestureMapper) -> симулятор
=============================================================================
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from .drone_link import DroneLink
from .gestures import (
    ACTION_NAMES,
    GestureConfig,
    GestureMapper,
    HOVER_COMMAND,
    DroneCommand,
    Posture,
)


#: Скільки секунд голосова команда (fly/sit) має переважати над жестами.
VOICE_HOLD_SECONDS = 1.0


class GestureDroneController:
    """
    Один виклик `update()` за кадр:

        controller = GestureDroneController(DroneLink())
        cmd = controller.update(assoc["owner_hands"], frame.shape)
        # cmd.label / cmd.to_text() -> у HUD
    """

    def __init__(
        self,
        link: DroneLink,
        mapper: Optional[GestureMapper] = None,
        enabled: bool = True,
    ) -> None:
        self.link = link
        self.mapper = mapper or GestureMapper(GestureConfig())
        self.enabled = enabled
        self.last_command: DroneCommand = HOVER_COMMAND
        self.last_update_at: Optional[float] = None
        self._voice: Optional[DroneCommand] = None
        self._voice_until: float = 0.0
        self._voice_lock = threading.Lock()

    # ------------------------------------------------------------------ режим
    def toggle(self) -> bool:
        """Увімкнути/вимкнути керування жестами (клавіша G)."""
        self.enabled = not self.enabled
        if not self.enabled:
            # одразу відпускаємо всі осі явним hover; далі DroneLink мовчить
            # (вектор застаріває за max_age) і симулятор тримає hover сам
            self.mapper.reset()
            self.last_command = HOVER_COMMAND
            self.link.send(HOVER_COMMAND)
        return self.enabled

    # ------------------------------------------------------------------ голос
    def voice_command(self, action: int, label: str = "", now: Optional[float] = None) -> None:
        """
        Голосова команда (fly = TAKEOFF, sit = LAND). Викликається з потоку STT.

        Утримується VOICE_HOLD_SECONDS: поки вона активна, update() шле саме її,
        а не HOVER з жестів (інакше жест перезаписав би команду за один кадр).
        """
        now = now if now is not None else time.monotonic()
        command = DroneCommand((0.0, 0.0, 0.0, 0.0, int(action)), Posture.NONE,
                               label or ACTION_NAMES.get(int(action), "VOICE"), "voice")
        with self._voice_lock:
            self._voice = command
            self._voice_until = now + VOICE_HOLD_SECONDS
        # відправляємо одразу, не чекаючи кадру CV (DroneLink.send — потокобезпечний)
        self.last_command = command
        self.link.send(command)

    def _active_voice(self, now: float) -> Optional[DroneCommand]:
        with self._voice_lock:
            if self._voice is not None and now < self._voice_until:
                return self._voice
            self._voice = None
            return None

    # ------------------------------------------------------------------ кадр
    def update(self, hands, frame_shape, now: Optional[float] = None) -> DroneCommand:
        """Розпізнає жест і (якщо увімкнено) ставить команду на відправку.

        `hands` — або список рук ({"points", "label"}), або словник результату
        MedusaEye (`assoc`) з ключем "owner_hands" — приймаються обидва варіанти.
        """
        owner_present = None
        if isinstance(hands, dict):
            owner_present = hands.get("owner_body") is not None
            hands = hands.get("owner_hands") or []

        now = now if now is not None else time.monotonic()
        # маппер оновлюємо завжди — щоб стан debounce жестів не застарів
        command = self.mapper.update(hands, frame_shape, now, owner_present=owner_present)
        self.last_update_at = now

        voice = self._active_voice(now)
        if voice is not None:
            # голосова команда (fly/sit) має пріоритет над жестами і діє навіть при G=OFF
            self.link.send(voice)
            self.last_command = voice
            return voice

        self.last_command = command
        if self.enabled:
            self.link.send(command)
        else:
            command = command.with_origin("disabled")
            self.last_command = command
        return command

    # ------------------------------------------------------------------ сервіс
    def reset(self) -> None:
        self.mapper.reset()
        self.last_command = HOVER_COMMAND

    def close(self) -> None:
        try:
            self.link.stop()
        except Exception:  # pragma: no cover - defensive
            pass

    def __enter__(self) -> "GestureDroneController":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
