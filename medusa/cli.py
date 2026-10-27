#!/usr/bin/env python3
"""
=============================================================================
  MedusaAlfa — CLI
  Спільний запуск для medusa_webcam.py та medusa_eye.py:
      CV (YOLO + MediaPipe) + жести + симулятор дрона, все однією командою
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .controller import GestureDroneController
from .drone_link import DroneLink
from .esp32_stream import DEFAULT_IP, DEFAULT_PORT
from .gestures import GESTURE_HELP, GestureConfig, GestureMapper, format_gesture_table
from .paths import ensure_project_dirs
from .sim_launcher import DEFAULT_HTTP_PORT, DEFAULT_SIM_HOST, DEFAULT_WS_PORT, SimLauncher

GESTURE_LIST = "\n".join(f"  {human:<32} {desc}" for _, human, desc in GESTURE_HELP)


def build_parser(default_source: str = "webcam") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "MedusaAlfa: розпізнавання хазяїна (YOLOv8-pose + face_recognition), жести рук "
            "(MediaPipe) та керування дроном у WebGL-симуляторі."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Жести (права рука, кількість розігнутих пальців):\n"
            f"{GESTURE_LIST}\n\n"
            "Додатково: зміщення руки ліворуч/праворуч від центру кадру = поворот (yaw).\n"
            "Клавіші у вікні відео: [G] жести ON/OFF, [R] реєстрація хазяїна, [D] debug,\n"
            "                     [C] скинути профіль, [Q]/[Esc] вихід\n\n"
            "Приклади:\n"
            "  python3 medusa_webcam.py\n"
            "  python3 medusa_webcam.py --no-sim --ws-url ws://127.0.0.1:8765\n"
            "  python3 medusa_eye.py --ip 10.52.127.86 --port 80\n"
        ),
    )

    src = parser.add_argument_group("камера")
    src.add_argument("--source", choices=["webcam", "esp32"], default=default_source,
                     help=f"джерело відео (за замовчуванням: {default_source})")
    src.add_argument("--camera", type=int, default=0, help="індекс локальної камери (default: 0)")
    src.add_argument("--ip", default=DEFAULT_IP, help=f"IP ESP32-CAM (default: {DEFAULT_IP})")
    src.add_argument("--port", type=int, default=DEFAULT_PORT,
                     help=f"TCP-порт ESP32-CAM (default: {DEFAULT_PORT})")

    sim = parser.add_argument_group("симулятор дрона")
    sim.add_argument("--no-sim", action="store_true", help="не запускати симулятор (він уже працює)")
    sim.add_argument("--no-browser", action="store_true", help="не відкривати браузер автоматично")
    sim.add_argument("--sim-host", default=DEFAULT_SIM_HOST, help=f"хост симулятора (default: {DEFAULT_SIM_HOST})")
    sim.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT,
                     help=f"HTTP-порт симулятора (default: {DEFAULT_HTTP_PORT})")
    sim.add_argument("--ws-port", type=int, default=DEFAULT_WS_PORT,
                     help=f"WebSocket-порт симулятора (default: {DEFAULT_WS_PORT})")
    sim.add_argument("--ws-url", default=None, help="готовий ws:// URL (перекриває --ws-port)")

    gst = parser.add_argument_group("жести та команди")
    gst.add_argument("--no-gestures", action="store_true", help="не керувати дроном (тільки CV)")
    gst.add_argument("--hand", choices=["Right", "Left", "any"], default="Right",
                     help="яка рука керує (default: Right)")
    gst.add_argument("--no-yaw-from-hand", action="store_true",
                     help="вимкнути поворот від зміщення руки в кадрі")
    gst.add_argument("--yaw-gain", type=float, default=55.0, help="°/с на максимальному відхиленні (default: 55)")
    gst.add_argument("--yaw-deadzone", type=float, default=0.22, help="мертва зона повороту 0..1 (default: 0.22)")
    gst.add_argument("--yaw-invert", action="store_true", help="інвертувати напрямок повороту")
    gst.add_argument("--mirrored", action="store_true",
                     help="кадр дзеркальний (selfie-режим); за замовчуванням webcam/ESP32 — недзеркальні")
    gst.add_argument("--approach-speed", type=float, default=1.5, help="м/с, жест V (default: 1.5)")
    gst.add_argument("--retreat-speed", type=float, default=1.2, help="м/с, жест три пальці (default: 1.2)")
    gst.add_argument("--climb-speed", type=float, default=0.9, help="м/с вгору (default: 0.9)")
    gst.add_argument("--descend-speed", type=float, default=0.9, help="м/с вниз (default: 0.9)")
    gst.add_argument("--stable-time", type=float, default=0.18,
                     help="с, скільки жест має протриматись (default: 0.18)")
    gst.add_argument("--auto-land", type=float, default=8.0,
                     help="с без хазяїна до автоматичного LAND, 0 = вимкнути (default: 8)")
    gst.add_argument("--link-rate", type=float, default=20.0, help="Гц відправки векторів (default: 20)")
    gst.add_argument("--gestures-table", action="store_true", help="показати таблицю жестів і вийти")
    return parser


def _gesture_config(args: argparse.Namespace) -> GestureConfig:
    return GestureConfig(
        approach_speed=args.approach_speed,
        retreat_speed=args.retreat_speed,
        climb_speed=args.climb_speed,
        descend_speed=args.descend_speed,
        yaw_from_hand=not args.no_yaw_from_hand,
        yaw_gain=args.yaw_gain,
        yaw_deadzone=args.yaw_deadzone,
        yaw_invert=args.yaw_invert,
        image_mirrored=args.mirrored,
        stable_time=args.stable_time,
        auto_land_after=args.auto_land,
        prefer_hand="" if args.hand == "any" else args.hand,
    )


def _banner(args, http_url: Optional[str], ws_url: str) -> None:
    print("=" * 72)
    print("  MEDUSA EYE  +  WEBGL DRONE SIM")
    print("=" * 72)
    print(f"  Камера           : {args.source}" + (f" (index {args.camera})" if args.source == "webcam"
                                                   else f" {args.ip}:{args.port}"))
    print(f"  Симулятор        : {http_url or 'не запускався (--no-sim)'}")
    print(f"  Керування        : {ws_url}")
    print(f"  Жести            : {'OFF (--no-gestures)' if args.no_gestures else 'ON'}"
          f" | рука: {args.hand} | стабільність {args.stable_time:.2f} c")
    print("-" * 72)
    print("  Жести: ✋ hover  ✊ land  ☝️ takeoff  ✌️ вперед  🤟 назад  👍 вгору  🤘 вниз")
    print("  Клавіші: [G] жести ON/OFF  [R] реєстрація  [D] debug  [C] скинути  [Q] вихід")
    print("=" * 72)


def main(argv=None, default_source: str = "webcam") -> int:
    """Точка входу для medusa_webcam.py / medusa_eye.py."""
    parser = build_parser(default_source)
    args = parser.parse_args(argv)

    if args.gestures_table:
        print("Жести (кількість розігнутих пальців правої руки):")
        print(format_gesture_table())
        return 0

    ensure_project_dirs()

    launcher: Optional[SimLauncher] = None
    link: Optional[DroneLink] = None
    controller: Optional[GestureDroneController] = None
    app = None

    ws_url = args.ws_url or f"ws://{args.sim_host}:{args.ws_port}"
    http_url: Optional[str] = None

    try:
        # 1. Симулятор (HTTP + WebSocket релей) -----------------------------------
        if not args.no_sim:
            launcher = SimLauncher(
                host=args.sim_host,
                http_port=args.http_port,
                ws_port=args.ws_port,
                open_browser=not args.no_browser,
            )
            status = launcher.ensure_running()
            http_url = status.http_url
            ws_url = args.ws_url or status.ws_url

        _banner(args, http_url, ws_url)

        # 2. Канал команд ---------------------------------------------------------
        link = DroneLink(ws_url, rate_hz=args.link_rate)
        link.start()

        # 3. Жести ---------------------------------------------------------------
        config = _gesture_config(args)
        controller = GestureDroneController(
            link,
            GestureMapper(config),
            enabled=not args.no_gestures,
        )

        # 4. Основний цикл CV ----------------------------------------------------
        from .eye import MedusaEye  # ліниво: важкі залежності (mediapipe, ultralytics)

        esp_endpoint = (args.ip, args.port) if args.source == "esp32" else None
        app = MedusaEye(
            source=args.source,
            esp32_url=esp_endpoint,
            controller=controller,
            camera_index=args.camera,
        )
        app.run()

        if link.connected:
            print(f"🔗 Відправлено векторів: {link.sent}")
        else:
            print(f"⚠ Зв'язок із симулятором не встановлено: {link.last_error or 'немає з’єднання'}")
        return 0

    except KeyboardInterrupt:
        print("\n⛔ Зупинено користувачем.")
        return 130
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 1
    finally:
        if controller is not None:
            controller.close()
        elif link is not None:
            link.stop()
        if launcher is not None:
            launcher.stop()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(default_source="webcam"))
