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
from .gestures import GestureConfig, GestureMapper, format_gesture_table
from .paths import ensure_project_dirs
from .sim_launcher import DEFAULT_HTTP_PORT, DEFAULT_SIM_HOST, DEFAULT_WS_PORT, SimLauncher
from .voice import VoiceCommander, voice_banner_text

GESTURE_LIST = format_gesture_table()


def build_parser(default_source: str = "webcam") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "MedusaAlfa: розпізнавання хазяїна (YOLOv8-pose + face_recognition), жести рук "
            "(MediaPipe) та керування дроном у WebGL-симуляторі."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Жести двома руками (кількість розігнутих пальців):\n"
            "  ЛІВА рука = команди (зліт/посадка/поворот), ПРАВА рука = рух.\n"
            f"{GESTURE_LIST}\n\n"
            "Жести обох рук комбінуються (напр. права 'вперед' + ліва 'поворот вправо').\n"
            "Клавіші у вікні відео: [G] жести ON/OFF, [R] реєстрація хазяїна, [D] debug,\n"
            "                     [C] скинути профіль, [Q]/[Esc] вихід\n\n"
            "Приклади:\n"
            "  python3 medusa_webcam.py\n"
            "  python3 medusa_webcam.py --no-sim --ws-url ws://127.0.0.1:8765\n"
            "  python3 medusa_eye.py --ip 10.21.246.241 --port 80\n"
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
    gst.add_argument("--swap-hands", action="store_true",
                     help="поміняти ролі рук (права = команди, ліва = рух) — для лівші")
    gst.add_argument("--mirrored", action="store_true",
                     help="кадр дзеркальний (selfie-режим); за замовчуванням webcam/ESP32 — недзеркальні")
    gst.add_argument("--forward-speed", type=float, default=1.5, help="м/с вперед (default: 1.5)")
    gst.add_argument("--backward-speed", type=float, default=1.2, help="м/с назад (default: 1.2)")
    gst.add_argument("--strafe-speed", type=float, default=1.2, help="м/с вліво/вправо (default: 1.2)")
    gst.add_argument("--yaw-speed", type=float, default=60.0, help="°/с поворот вліво/вправо (default: 60)")
    gst.add_argument("--stable-time", type=float, default=0.18,
                     help="с, скільки жест має протриматись (default: 0.18)")
    gst.add_argument("--auto-land", type=float, default=8.0,
                     help="с без хазяїна до автоматичного LAND, 0 = вимкнути (default: 8)")
    gst.add_argument("--link-rate", type=float, default=20.0, help="Гц відправки векторів (default: 20)")
    gst.add_argument("--gestures-table", action="store_true", help="показати таблицю жестів і вийти")

    voc = parser.add_argument_group("голосові команди (ElevenLabs)")
    voc.add_argument("--no-voice", action="store_true",
                     help="вимкнути голосові команди SPACE + fly/sit")
    voc.add_argument("--voice-lang", default="en",
                     help="мова розпізнавання ElevenLabs, ISO-639 код; порожньо = авто (default: en)")
    return parser


def _gesture_config(args: argparse.Namespace) -> GestureConfig:
    return GestureConfig(
        forward_speed=args.forward_speed,
        backward_speed=args.backward_speed,
        strafe_speed=args.strafe_speed,
        yaw_speed=args.yaw_speed,
        image_mirrored=args.mirrored,
        swap_hands=args.swap_hands,
        stable_time=args.stable_time,
        auto_land_after=args.auto_land,
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
          f" | стабільність {args.stable_time:.2f} c")
    print("-" * 72)
    cmd_hand, move_hand = ("ПРАВА", "ЛІВА") if args.swap_hands else ("ЛІВА", "ПРАВА")
    print(f"  {cmd_hand} рука (команди): ☝️ зліт  ✊ посадка  ✌️ поворот вліво  ✋ поворот вправо")
    print(f"  {move_hand} рука (рух)   : ☝️ вперед  ✌️ назад  ✊ вліво  ✋ вправо")
    print("  Клавіші: [G] жести ON/OFF  [R] реєстрація  [D] debug  [C] скинути  [Q] вихід")
    print(f"  Голос            : {voice_banner_text(args.no_voice)}")
    print("=" * 72)


def main(argv=None, default_source: str = "webcam") -> int:
    """Точка входу для medusa_webcam.py / medusa_eye.py."""
    parser = build_parser(default_source)
    args = parser.parse_args(argv)

    if args.gestures_table:
        print("Жести двома руками (кількість розігнутих пальців):")
        print(format_gesture_table(_gesture_config(args)))
        return 0

    ensure_project_dirs()

    launcher: Optional[SimLauncher] = None
    link: Optional[DroneLink] = None
    controller: Optional[GestureDroneController] = None
    voice_cmd: Optional[VoiceCommander] = None
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

        # 3b. Голос (push-to-talk SPACE: fly / sit) ----------------------------------
        if not args.no_voice:
            voice = VoiceCommander.from_env(controller, language_code=args.voice_lang or None)
            voice.start()
            voice_cmd = voice

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
        if voice_cmd is not None:
            voice_cmd.stop()
        if controller is not None:
            controller.close()
        elif link is not None:
            link.stop()
        if launcher is not None:
            launcher.stop()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(default_source="webcam"))
