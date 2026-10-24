#!/usr/bin/env python3
"""
=============================================================================
  MedusaAlfa — Medusa Eye (WEBCAM версія)  +  DRONE SIM
=============================================================================
Запускає ВСЕ однією командою:
    1) WebGL-симулятор дрона на http://127.0.0.1:8000 (відкривається браузер);
    2) розпізнавання хазяїна (YOLOv8-pose + face_recognition) з локальної камери;
    3) трекінг рук (MediaPipe) і перетворення жестів у вектор команд дрона
       [v_x, v_y, v_z, yaw_rate, action] -> WebSocket -> симулятор.

Запуск:
    python3 medusa_webcam.py
    python3 medusa_webcam.py --no-sim            # симулятор уже запущений окремо
    python3 medusa_webcam.py --no-gestures       # тільки CV, без керування
    python3 medusa_webcam.py --gestures-table    # показати таблицю жестів

Клавіші у вікні відео: [G] жести ON/OFF, [R] реєстрація хазяїна, [D] debug,
                      [C] скинути профіль, [Q]/[Esc] вихід.
=============================================================================
"""

import sys

from medusa.cli import main

if __name__ == "__main__":
    sys.exit(main(default_source="webcam"))

