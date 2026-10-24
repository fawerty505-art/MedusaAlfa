#!/usr/bin/env python3
"""
=============================================================================
  MedusaAlfa — Overlay
  HUD поверх відео: розпізнаний жест, вектор команд, стан зв'язку з симулятором
=============================================================================
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2

from .gestures import DroneCommand, ACTION_NAMES

COLOR_OK = (0, 220, 90)
COLOR_WARN = (0, 200, 255)
COLOR_BAD = (0, 0, 235)
COLOR_TEXT = (225, 228, 235)
COLOR_MUTED = (150, 156, 168)
COLOR_ACCENT = (0, 220, 255)
PANEL_BG = (24, 26, 31)

PANEL_WIDTH = 330
LINE_HEIGHT = 16


def _put(image, text: str, x: int, y: int, color, scale: float = 0.38, thickness: int = 1) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _panel(image, x: int, y: int, width: int, height: int, alpha: float = 0.72) -> None:
    """Напівпрозора панель (щоб текст читався на будь-якому фоні)."""
    h, w = image.shape[:2]
    x2 = min(w, x + width)
    y2 = min(h, y + height)
    x = max(0, x)
    y = max(0, y)
    if x2 <= x or y2 <= y:
        return
    region = image[y:y2, x:x2]
    backdrop = region.copy()
    backdrop[:] = PANEL_BG
    cv2.addWeighted(backdrop, alpha, region, 1.0 - alpha, 0, region)
    cv2.rectangle(image, (x, y), (x2, y2), (70, 76, 88), 1, cv2.LINE_AA)


def _telemetry_line(telemetry: Optional[Dict[str, Any]]) -> str:
    if not telemetry:
        return "SIM     : очікування телеметрії..."
    status = str(telemetry.get("status", "?"))
    alt = telemetry.get("y")
    heading = telemetry.get("heading")
    parts = [f"SIM     : {status}"]
    if isinstance(alt, (int, float)):
        parts.append(f"ALT {max(0.0, float(alt)):.2f} m")
    if isinstance(heading, (int, float)):
        parts.append(f"HDG {float(heading):.0f}°")
    return "  ".join(parts)


def draw_drone_panel(
    frame,
    command: Optional[DroneCommand] = None,
    link=None,
    enabled: bool = True,
    telemetry: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Малює панель керування дроном у правому нижньому куті кадру.

    frame      — BGR-кадр (змінюється на місці);
    command    — остання DroneCommand (або None);
    link       — DroneLink (має .status_text() та .connected), може бути None;
    enabled    — чи увімкнено керування жестами;
    telemetry  — словник телеметрії від симулятора.
    """
    h, w = frame.shape[:2]
    lines = 6
    height = lines * LINE_HEIGHT + 16
    x = max(0, w - PANEL_WIDTH - 10)
    y = max(0, h - height - 30)      # 30 px — місце під нижньою смугою клавіш
    _panel(frame, x, y, PANEL_WIDTH, height)

    tx = x + 8
    ty = y + 16

    _put(frame, "GESTURE FLIGHT CONTROL", tx, ty, COLOR_ACCENT, 0.40, 1)
    ty += LINE_HEIGHT

    state_text = "ON" if enabled else "OFF  (натисніть G)"
    _put(frame, "CONTROL : ", tx, ty, COLOR_MUTED)
    _put(frame, state_text, tx + 68, ty, COLOR_OK if enabled else COLOR_WARN)
    ty += LINE_HEIGHT

    if command is None:
        _put(frame, "GESTURE : —", tx, ty, COLOR_MUTED)
        ty += LINE_HEIGHT
        _put(frame, "VECTOR  : —", tx, ty, COLOR_MUTED)
    else:
        gesture_color = COLOR_WARN if not enabled else COLOR_TEXT
        _put(frame, "GESTURE : ", tx, ty, COLOR_MUTED)
        _put(frame, f"{command.posture} -> {command.label}", tx + 68, ty, gesture_color)
        ty += LINE_HEIGHT
        _put(frame, "VECTOR  : ", tx, ty, COLOR_MUTED)
        _put(frame, command.to_text(), tx + 68, ty,
             COLOR_ACCENT if command.action == 0 else COLOR_WARN)
    ty += LINE_HEIGHT

    if link is not None:
        status = link.status_text()
        color = COLOR_OK if getattr(link, "connected", False) else COLOR_BAD
        _put(frame, "LINK    : ", tx, ty, COLOR_MUTED)
        _put(frame, f"{status}  {getattr(link, 'url', '')}", tx + 68, ty, color)
    else:
        _put(frame, "LINK    : вимкнено (--no-link)", tx, ty, COLOR_BAD)
    ty += LINE_HEIGHT

    _put(frame, _telemetry_line(telemetry), tx, ty, COLOR_MUTED, 0.36)


def draw_action_flash(frame, command: Optional[DroneCommand], now: float, last_action_at: Optional[float]) -> None:
    """Коротке велике підтвердження дії (TAKEOFF / LAND) по центру кадру."""
    if command is None or command.action == 0 or last_action_at is None:
        return
    if now - last_action_at > 0.9:
        return
    h, w = frame.shape[:2]
    text = ACTION_NAMES.get(command.action, "ACTION")
    color = COLOR_OK if command.action == 1 else COLOR_WARN
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    x = (w - tw) // 2
    y = int(h * 0.20)
    _panel(frame, x - 16, y - th - 10, tw + 32, th + 22, alpha=0.6)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
