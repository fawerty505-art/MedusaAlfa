#!/usr/bin/env python3
"""
=============================================================================
  MedusaAlfa — Medusa Eye (ESP32-CAM версія)  +  DRONE SIM
=============================================================================
Той самий движок, що й medusa_webcam.py (YOLOv8-pose, MediaPipe руки,
розпізнавання хазяїна, жести, симулятор дрона), але відео береться з ESP32-CAM
по TCP (4-байтна довжина кадру + JPEG).

Запуск:
    python3 medusa_eye.py                          # ESP32 за замовчуванням
    python3 medusa_eye.py --ip 10.21.246.241 --port 80
    python3 medusa_eye.py --source webcam           # те саме, що medusa_webcam.py
    python3 medusa_eye.py --no-sim                  # без запуску симулятора

Клавіші у вікні відео: [G] жести ON/OFF, [R] реєстрація хазяїна, [D] debug,
                      [C] скинути профіль, [Q]/[Esc] вихід.
=============================================================================
"""

import sys

from medusa.cli import main

if __name__ == "__main__":
    sys.exit(main(default_source="esp32"))
