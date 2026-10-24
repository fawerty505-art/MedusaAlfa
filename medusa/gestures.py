#!/usr/bin/env python3
"""
=============================================================================
  MedusaAlfa — Gesture Engine
  Жести рук (MediaPipe, 21 точка)  ->  вектор команд дрона
=============================================================================

Симулятор (sim/app.js) очікує рівно такий вектор:

    [v_x, v_y, v_z, yaw_rate, action]

    v_x       m/s   вперед(+) / назад(-)   — у системі координат КОРПУСУ дрона
    v_y       m/s   вправо(+) / вліво(-)   — у системі координат КОРПУСУ дрона
    v_z       m/s   вгору(+) / вниз(-)
    yaw_rate  °/s   додатне = поворот за годинниковою стрілкою (вправо)
    action    0 HOVER | 1 TAKEOFF (на 2 м) | 2 LAND | 3 RESET

Таблиця жестів (розпізнається кількість розігнутих пальців — це стійко
до повороту руки, освітлення та відстані):

    ┌───────────────────────────────┬──────────────┬───────────────────────────┐
    │ Жест                          │ Вектор       │ Дія дрона                 │
    ├───────────────────────────────┼──────────────┼───────────────────────────┤
    │ ✋ Долоня (4 пальці + великий) │ [0,0,0,0,0]  │ HOVER — тримати позицію   │
    │ ✊ Кулак (0)                   │ [0,0,0,0,2]  │ LAND — сісти             │
    │ ☝️ Вказівний (1)               │ [0,0,0,0,1]  │ TAKEOFF — злетіти на 2 м  │
    │ ✌️ V / два (2)                 │ [+vx,0,0,0,0]│ FORWARD — наблизитися     │
    │ 🤟 Три (3)                     │ [-vx,0,0,0,0]│ BACKWARD — відлетіти назад│
    │ 👍 Тільки великий палець       │ [0,0,+vz,0,0]│ CLIMB — вище             │
    │ 🖖 Чотири (без великого)       │ [0,0,-vz,0,0]│ DESCEND — нижче           │
    └───────────────────────────────┴──────────────┴───────────────────────────┘

Додатково (безперервне керування): якщо рука зміщена ліворуч/праворуч від
центру кадру, додається yaw_rate — дрон плавно повертається в той бік.
Це працює для "польотних" жестів (долоня, V, три, чотири, палець) і не
впливає на жести-команди (кулак, вказівний).

Захист від випадкових команд:
    * жест має протриматись `stable_time` секунд (анти-дриґання);
    * gesture-команда надсилається з частотою до 20 Гц (симулятор має
      власний 0.5-секундний safety-timeout -> автоперехід у hover);
    * якщо хазяїн зник з кадру назавжди довше `auto_land_after` секунд —
      надсилається LAND (можна вимкнути, auto_land_after=0).

Модуль НЕ залежить від OpenCV/MediaPipe/numpy — тому його легко тестувати
(див. tests/test_gestures.py): на вхід подаються 21 точка кисті.
=============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Індекси точок MediaPipe Hands (21 landmark)
# ─────────────────────────────────────────────────────────────────────────────
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
HAND_LANDMARKS = 21

#: (PIP-суглоб, кінчик) для чотирьох пальців
FINGER_CHAINS = (
    (INDEX_PIP, INDEX_TIP),
    (MIDDLE_PIP, MIDDLE_TIP),
    (RING_PIP, RING_TIP),
    (PINKY_PIP, PINKY_TIP),
)
#: точки "долоні" — їх середнє дає стабільний центр кисті
PALM_POINTS = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

#: Дії симулятора
HOVER, TAKEOFF, LAND, RESET = 0, 1, 2, 3
ACTION_NAMES = {HOVER: "HOVER", TAKEOFF: "TAKEOFF", LAND: "LAND", RESET: "RESET"}

Point = Tuple[float, float]


class Posture(str, Enum):
    """Розпізнана поза кисті."""

    NONE = "none"
    FIST = "fist"
    POINT = "point"
    VICTORY = "victory"
    THREE = "three"
    FOUR = "four"
    PALM = "palm"
    THUMB = "thumb"

    def __str__(self) -> str:  # зручно друкувати в HUD
        return self.value.upper()

# ─────────────────────────────────────────────────────────────────────────────
# Геометрія кисті
# ─────────────────────────────────────────────────────────────────────────────
def _point(value) -> Point:
    """Приймає (x, y) tuple/list/np.ndarray і повертає звичайний tuple float."""
    x, y = value[0], value[1]
    return float(x), float(y)


def distance(a, b) -> float:
    ax, ay = _point(a)
    bx, by = _point(b)
    return math.hypot(ax - bx, ay - by)


def hand_scale(points: Sequence) -> float:
    """Масштаб кисті в пікселях: відстань зап'ястя до середнього пальця (MCP)."""
    return max(distance(points[WRIST], points[MIDDLE_MCP]), 1e-6)


def normalize_hand(points) -> Optional[List[Point]]:
    """Перевіряє та нормалізує 21 точку кисті; None якщо дані некоректні."""
    if points is None:
        return None
    try:
        pts = [_point(p) for p in points]
    except (TypeError, ValueError, IndexError):
        return None
    return pts if len(pts) >= HAND_LANDMARKS else None


def fingers_up(points: Sequence, tip_ratio: float = 1.15, thumb_ratio: float = 1.15):
    """
    Повертає (thumb, index, middle, ring, pinky) — True якщо палець розігнутий.

    Метод інваріантний до повороту кисті: кінчик пальця має бути помітно далі
    від зап'ястя, ніж відповідний PIP-суглоб (у зігнутому пальці кінчик
    "завертається" до долоні й опиняється ближче до зап'ястя).
    Великий палець додатково перевіряється на відстань від основи вказівного —
    інакше він виглядає "розігнутим" навіть у кулаці.
    """
    wrist = points[WRIST]
    fingers = []
    for pip, tip in FINGER_CHAINS:
        fingers.append(distance(points[tip], wrist) > distance(points[pip], wrist) * tip_ratio)

    thumb_tip = points[THUMB_TIP]
    thumb_extended = (
        distance(thumb_tip, wrist) > distance(points[THUMB_IP], wrist) * thumb_ratio
        and distance(thumb_tip, points[INDEX_MCP]) > 0.8 * hand_scale(points)
    )
    return (thumb_extended, *fingers)


def count_fingers(points) -> int:
    """Кількість розігнутих пальців (0..5), великий палець включно."""
    return sum(1 for up in fingers_up(points) if up)


def classify_posture(points) -> Posture:
    """Класифікує позу кисті за розігнутими пальцями."""
    pts = normalize_hand(points)
    if pts is None:
        return Posture.NONE

    thumb, index, middle, ring, pinky = fingers_up(pts)
    count = sum((index, middle, ring, pinky))

    if count == 4:
        # долоня = чотири пальці + відставлений великий; "чотири" = без великого
        return Posture.PALM if thumb else Posture.FOUR
    if count == 0:
        return Posture.THUMB if thumb else Posture.FIST
    if count == 1:
        return Posture.POINT if index else Posture.NONE
    if count == 2:
        return Posture.VICTORY if (index and middle) else Posture.NONE
    if count == 3:
        return Posture.THREE if (index and middle and ring) else Posture.NONE
    return Posture.NONE


def palm_center(points) -> Point:
    """Центр долоні — стабільніша точка за зап'ястя (менше тремтить)."""
    pts = [_point(points[i]) for i in PALM_POINTS]
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


# ─────────────────────────────────────────────────────────────────────────────
# Команда дрона
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DroneCommand:
    """Готова команда для симулятора: [v_x, v_y, v_z, yaw_rate, action]."""

    vector: Tuple[float, float, float, float, int]
    posture: Posture = Posture.NONE
    label: str = "HOLD"
    origin: str = "gesture"          # gesture | no-hand | auto-land | disabled

    @property
    def action(self) -> int:
        return int(self.vector[4])

    @property
    def action_name(self) -> str:
        return ACTION_NAMES.get(self.action, str(self.action))

    def to_payload(self) -> List[float]:
        """JSON-сумісний список для WebSocket (action завжди ціле)."""
        vx, vy, vz, yaw, action = self.vector
        return [round(float(vx), 4), round(float(vy), 4), round(float(vz), 4),
                round(float(yaw), 4), int(action)]

    def to_text(self) -> str:
        vx, vy, vz, yaw, action = self.to_payload()
        return f"[{vx:+.2f}, {vy:+.2f}, {vz:+.2f}, {yaw:+.1f}, {action}]"

    def with_origin(self, origin: str) -> "DroneCommand":
        return DroneCommand(self.vector, self.posture, self.label, origin)


HOVER_COMMAND = DroneCommand((0.0, 0.0, 0.0, 0.0, HOVER), Posture.NONE, "HOLD")



# ─────────────────────────────────────────────────────────────────────────────
# Конфігурація та таблиця жестів
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GestureConfig:
    """Налаштування жестів (усі швидкості — у тих самих одиницях, що й вектор)."""

    approach_speed: float = 1.5      # м/с вперед (жест V)
    retreat_speed: float = 1.2       # м/с назад (жест "три")
    climb_speed: float = 0.9         # м/с вгору (жест "чотири")
    descend_speed: float = 0.9       # м/с вниз  (жест "великий палець")

    yaw_from_hand: bool = True       # зміщення руки в кадрі -> yaw_rate
    yaw_gain: float = 55.0           # °/с на максимальному відхиленні
    yaw_deadzone: float = 0.22       # |offset| < deadzone -> без повороту
    yaw_invert: bool = False         # поміняти напрямок повороту

    stable_time: float = 0.18        # с, скільки жест має протриматись
    hand_lost_timeout: float = 0.35  # с, ігнорування "блимнув" руки
    auto_land_after: float = 8.0     # с без хазяїна -> LAND (0 = вимкнути)
    prefer_hand: str = "Right"       # "Right" | "Left" | "" (будь-яка)

    def __post_init__(self) -> None:
        if self.stable_time < 0:
            self.stable_time = 0.0
        if self.hand_lost_timeout < 0:
            self.hand_lost_timeout = 0.0
        if self.auto_land_after < 0:
            self.auto_land_after = 0.0
        self.yaw_deadzone = min(max(self.yaw_deadzone, 0.0), 0.95)
        self.yaw_gain = max(self.yaw_gain, 0.0)


@dataclass(frozen=True)
class GestureBinding:
    """Що робить дрон, коли розпізнано позу."""

    label: str
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    action: int = HOVER
    allow_yaw: bool = True

    def as_command(self, posture: Posture, yaw_rate: float = 0.0) -> DroneCommand:
        yaw = yaw_rate if self.allow_yaw else 0.0
        return DroneCommand(
            (round(float(self.vx), 4), round(float(self.vy), 4), round(float(self.vz), 4),
             round(float(yaw), 4), int(self.action)),
            posture,
            self.label,
        )


def default_bindings(config: Optional[GestureConfig] = None) -> Dict[Posture, GestureBinding]:
    """Стандартна таблиця жестів -> команди (її і можна перевизначати)."""
    cfg = config or GestureConfig()
    return {
        Posture.PALM: GestureBinding("HOVER", action=HOVER),
        Posture.FIST: GestureBinding("LAND", action=LAND, allow_yaw=False),
        Posture.POINT: GestureBinding("TAKEOFF", action=TAKEOFF, allow_yaw=False),
        Posture.VICTORY: GestureBinding("FORWARD / APPROACH", vx=cfg.approach_speed),
        Posture.THREE: GestureBinding("BACKWARD / RETREAT", vx=-cfg.retreat_speed),
        Posture.THUMB: GestureBinding("CLIMB UP", vz=cfg.climb_speed),
        Posture.FOUR: GestureBinding("DESCEND DOWN", vz=-cfg.descend_speed),
    }


#: Порядок для HUD/README
GESTURE_HELP = (
    (Posture.PALM, "Долоня (4 пальці + великий)", "HOVER — тримати позицію"),
    (Posture.FIST, "Кулак (0 пальців)", "LAND — сісти"),
    (Posture.POINT, "Один палець (вказівний)", "TAKEOFF — злетіти на 2 м"),
    (Posture.VICTORY, "Два пальці (V)", "FORWARD — наблизитися до вас"),
    (Posture.THREE, "Три пальці", "BACKWARD — відлетіти назад"),
    (Posture.THUMB, "Тільки великий палець (👍)", "CLIMB — піднятися вище"),
    (Posture.FOUR, "Чотири пальці (без великого)", "DESCEND — опуститися нижче"),
)



# ─────────────────────────────────────────────────────────────────────────────
# Основний маппер: кадр -> команда
# ─────────────────────────────────────────────────────────────────────────────
class GestureMapper:
    """
    Перетворює руки хазяїна (вихід MediaPipe) у команду дрона.

    Використання у циклі відео:

        mapper = GestureMapper()
        cmd = mapper.update(assoc["owner_hands"], frame.shape, time.time())
        ws.send(cmd.to_payload())

    `hands` — список словників {"points": [(x, y) * 21], "label": "Right"|"Left"}.
    """

    def __init__(
        self,
        config: Optional[GestureConfig] = None,
        bindings: Optional[Dict[Posture, GestureBinding]] = None,
    ) -> None:
        self.config = config or GestureConfig()
        self.bindings = bindings if bindings is not None else default_bindings(self.config)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Скидає внутрішній стан (наприклад, після реєстрації нового хазяїна)."""
        self._candidate = Posture.NONE
        self._candidate_since: Optional[float] = None
        self._active = Posture.NONE
        self._last_hand_time: Optional[float] = None
        self._last_hand_label = ""
        self._auto_landed = False
        self._yaw_state = 0.0
        self._last_yaw_time: Optional[float] = None
        self.last_command = HOVER_COMMAND

    # ------------------------------------------------------------------ руки
    def _select_hand(self, hands) -> Optional[dict]:
        """Обирає руку для керування: спершу потрібну (права), потім будь-яку з жестом."""
        prepared = []
        for hand in hands or ():
            pts = normalize_hand(hand.get("points"))
            if pts is None:
                continue
            prepared.append({"points": pts, "label": str(hand.get("label") or "")})

        if not prepared:
            return None

        preferred = self.config.prefer_hand
        if preferred:
            for hand in prepared:
                if hand["label"].lower() == preferred.lower():
                    return hand

        # немає потрібної руки — беремо першу, де взагалі видно жест
        for hand in prepared:
            if classify_posture(hand["points"]) is not Posture.NONE:
                return hand
        return prepared[0]

    # ------------------------------------------------------------------ yaw
    def _yaw_rate(self, points: Sequence[Point], frame_width: float) -> float:
        """Поворот корпусу від горизонтального зміщення руки в кадрі."""
        cfg = self.config
        if not cfg.yaw_from_hand or cfg.yaw_gain <= 0 or frame_width <= 0:
            return 0.0

        cx, _ = palm_center(points)
        offset = (cx - frame_width / 2.0) / (frame_width / 2.0)   # -1 .. +1
        offset = max(-1.0, min(1.0, offset))
        magnitude = abs(offset)
        if magnitude <= cfg.yaw_deadzone:
            return 0.0

        scaled = (magnitude - cfg.yaw_deadzone) / (1.0 - cfg.yaw_deadzone)
        sign = 1.0 if offset > 0 else -1.0
        if cfg.yaw_invert:
            sign = -sign
        return sign * scaled * cfg.yaw_gain


    # ------------------------------------------------------------------ update
    def update(self, hands, frame_shape, now: Optional[float] = None) -> DroneCommand:
        """Головна функція: руки + розмір кадру + час -> DroneCommand."""
        now = now if now is not None else _now()
        cfg = self.config

        width = float(frame_shape[1]) if frame_shape and len(frame_shape) >= 2 else 0.0

        hand = self._select_hand(hands)
        if hand is None:
            return self._no_hand_command(now)

        # рука є -> авто-лендінг скасовуємо
        self._last_hand_time = now
        self._auto_landed = False
        self._last_hand_label = hand["label"]

        posture = classify_posture(hand["points"])

        # анти-дриґання: жест стає активним лише після stable_time
        if posture is not self._candidate:
            self._candidate = posture
            self._candidate_since = now
        if self._candidate_since is not None and (now - self._candidate_since) >= cfg.stable_time:
            self._active = self._candidate

        active = self._active
        binding = self.bindings.get(active)

        if binding is None:
            command = DroneCommand((0.0, 0.0, 0.0, 0.0, HOVER), active, "HOLD (no gesture)")
            self.last_command = command
            return command

        yaw_rate = 0.0
        if binding.allow_yaw:
            yaw_rate = self._smooth_yaw(self._yaw_rate(hand["points"], width), now)

        command = binding.as_command(active, yaw_rate)
        self.last_command = command
        return command

    def _smooth_yaw(self, target: float, now: float) -> float:
        """Експоненційне згладжування yaw_rate (щоб дрон не смикався)."""
        dt = 0.0 if self._last_yaw_time is None else max(0.0, min(now - self._last_yaw_time, 0.5))
        self._last_yaw_time = now
        alpha = 1.0 if dt <= 0 else 1.0 - math.exp(-dt / 0.15)
        self._yaw_state += (target - self._yaw_state) * alpha
        return self._yaw_state

    def _no_hand_command(self, now: float) -> DroneCommand:
        """Руки немає: тримаємо hover, а через auto_land_after секунд — LAND."""
        cfg = self.config
        self._candidate = Posture.NONE
        self._candidate_since = None
        self._active = Posture.NONE
        self._yaw_state = 0.0
        self._last_yaw_time = None

        lost_for = None if self._last_hand_time is None else now - self._last_hand_time
        if cfg.auto_land_after > 0 and lost_for is not None and lost_for >= cfg.auto_land_after:
            self._auto_landed = True
            command = DroneCommand((0.0, 0.0, 0.0, 0.0, LAND), Posture.NONE, "AUTO-LAND", "auto-land")
        else:
            origin = "no-hand" if lost_for is not None else "idle"
            command = DroneCommand((0.0, 0.0, 0.0, 0.0, HOVER), Posture.NONE, "NO HAND", origin)
        self.last_command = command
        return command


def _now() -> float:
    """Монотонний час у секундах."""
    import time

    return time.monotonic()


def describe_command(command: Optional[DroneCommand]) -> str:
    """Короткий опис команди для логів/HUD."""
    if command is None:
        return "—"
    return f"{command.label} {command.to_text()}"


def format_gesture_table(bindings: Optional[Dict[Posture, GestureBinding]] = None) -> str:
    """Текстова таблиця жестів (для --help та README)."""
    bindings = bindings or default_bindings()
    lines = []
    for posture, human, _ in GESTURE_HELP:
        binding = bindings.get(posture)
        vector = binding.as_command(posture).to_text() if binding else "—"
        lines.append(f"  {human:<30} {vector:<34} {binding.label if binding else '—'}")
    return "\n".join(lines)

