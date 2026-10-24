"""
MedusaAlfa — комп'ютерний зір (YOLOv8-pose + MediaPipe) + керування дроном жестами.

Пакет об'єднує:
    * eye.py          — розпізнавання хазяїна (обличчя + тіло) та рук (MediaPipe)
    * gestures.py     — перетворення жестів у вектор команд [v_x, v_y, v_z, yaw_rate, action]
    * drone_link.py   — WebSocket-клієнт до симулятора дрона
    * sim_launcher.py — запуск локального симулятора (sim/server.py) разом з програмою
    * cli.py          — спільний командний рядок для medusa_webcam.py / medusa_eye.py
"""

__all__ = [
    "eye",
    "esp32_stream",
    "gestures",
    "drone_link",
    "controller",
    "sim_launcher",
    "overlay",
    "paths",
    "cli",
]

__version__ = "1.0.0"
