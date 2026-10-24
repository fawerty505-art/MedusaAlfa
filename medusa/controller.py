#!/usr/bin/env python3
"""
=============================================================================
  MedusaAlfa — Gesture Drone Controller
  Склеює три частини: руки (MediaPipe) -> жест (GestureMapper) -> симулятор
=============================================================================
"""

from __future__ import annotations

import time
from typing import Optional

from .drone_link import DroneLink
from .gestures import (
    GestureConfig,
    GestureMapper,
    HOVER_COMMAND,
    DroneCommand,
)


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

    # ------------------------------------------------------------------ режим
    def toggle(self) -> bool:
        """Увімкнути/вимкнути керування жестами (клавіша G)."""
        self.enabled = not self.enabled
        if not self.enabled:
            # при вимкненні одразу "відпускаємо" всі осі:
            # далі симулятор сам плавно перейде в hover (safety-timeout 0.5 с)
            self.mapper.reset()
        return self.enabled

    # ------------------------------------------------------------------ кадр
    def update(self, hands, frame_shape, now: Optional[float] = None) -> DroneCommand:
        """Розпізнає жест і (якщо увімкнено) ставить команду на відправку.

        `hands` — або список рук ({"points", "label"}), або словник результату
        MedusaEye (`assoc`) з ключем "owner_hands" — приймаються обидва варіанти.
        """
        if isinstance(hands, dict):
            hands = hands.get("owner_hands") or []

        now = now if now is not None else time.monotonic()
        command = self.mapper.update(hands, frame_shape, now)
        self.last_command = command
        self.last_update_at = now

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
