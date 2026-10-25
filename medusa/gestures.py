#!/usr/bin/env python3
"""
=============================================================================
  MedusaAlfa — Gesture Engine
  Жести ДВОХ рук (MediaPipe, 21 точка)  ->  вектор команд дрона
=============================================================================

Симулятор (sim/app.js) очікує рівно такий вектор:

    [v_x, v_y, v_z, yaw_rate, action]

    v_x       m/s   вперед(+) / назад(-)   — у системі координат КОРПУСУ дрона
    v_y       m/s   вправо(+) / вліво(-)   — у системі координат КОРПУСУ дрона
    v_z       m/s   вгору(+) / вниз(-)
    yaw_rate  °/s   додатне = поворот за годинниковою стрілкою (вправо)
    action    0 HOVER | 1 TAKEOFF (на 2 м) | 2 LAND | 3 RESET

Керування двома руками (розпізнається кількість розігнутих пальців — це
стійко до повороту руки, освітлення та відстані):

    ЛІВА рука — КОМАНДИ (зліт / посадка / поворот), 4 жести
    ┌──────────────────────┬────────────────────────────┐
    │ ☝️ 1 палець           │ TAKEOFF — злетіти на 2 м    │
    │ ✊ кулак              │ LAND — сісти                │
    │ ✌️ 2 пальці           │ TURN LEFT  — поворот вліво  │
    │ ✋ долоня             │ TURN RIGHT — поворот вправо │
    └──────────────────────┴────────────────────────────┘

    ПРАВА рука — РУХ (вперед / назад / вліво / вправо), 4 жести
    ┌──────────────────────┬────────────────────────────┐
    │ ☝️ 1 палець           │ FORWARD  — вперед           │
    │ ✌️ 2 пальці           │ BACKWARD — назад            │
    │ ✊ кулак              │ LEFT     — вліво            │
    │ ✋ долоня             │ RIGHT    — вправо           │
    └──────────────────────┴────────────────────────────┘

Інші пози (3 пальці, "рок" тощо) нічого не роблять. Якщо рука не в кадрі або
показує невідомий жест — вона просто нічого не додає до команди.

Жести обох рук комбінуються: права "вперед" + ліва "поворот вправо" = політ
дугою. Дії TAKEOFF/LAND мають пріоритет — під час них рух ігнорується.
Великий палець не враховується: кулак лишається кулаком, навіть якщо великий
палець стирчить, а долоня — долонею, навіть якщо він притиснутий.

Яка рука ліва/права — з погляду КОРИСТУВАЧА. MediaPipe вважає кадр
дзеркальним, а наші джерела (webcam без flip, ESP32) — недзеркальні, тому
мітки автоматично міняються місцями (image_mirrored=False). Якщо обидві
руки отримали однакову мітку, сторона визначається за позицією в кадрі.
Для лівші: swap_hands=True (CLI: --swap-hands) міняє ролі рук місцями.

Захист від випадкових команд:
    * жест кожної руки має протриматись `stable_time` секунд (анти-дриґання);
    * короткі пропадання руки (< `hand_lost_timeout`) ігноруються;
    * команда надсилається з частотою 20 Гц (симулятор має власний
      0.5-секундний safety-timeout -> автоперехід у hover);
    * якщо ХАЗЯЇН (не рука!) зник з кадру довше `auto_land_after` секунд —
      надсилається LAND (auto_land_after=0 вимикає). Якщо хазяїн у кадрі,
      а рук не видно — дрон просто висить у hover.

Модуль НЕ залежить від OpenCV/MediaPipe/numpy — тому його легко тестувати
(див. tests/test_gestures.py): на вхід подаються 21 точка кисті.
=============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

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

#: Ролі рук
ROLE_COMMAND = "command"   # зліт / посадка / поворот        (за замовчуванням ЛІВА рука)
ROLE_MOVE = "move"         # вперед / назад / вліво / вправо (за замовчуванням ПРАВА рука)
ROLES = (ROLE_COMMAND, ROLE_MOVE)

Point = Tuple[float, float]


class Posture(str, Enum):
    """Розпізнана поза кисті."""

    NONE = "none"
    FIST = "fist"
    POINT = "point"
    VICTORY = "victory"
    THREE = "three"
    ROCK = "rock"
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
        return Posture.PALM            # великий палець навмисно не враховується
    if count == 0:
        return Posture.THUMB if thumb else Posture.FIST
    if count == 1:
        return Posture.POINT if index else Posture.NONE
    if count == 2:
        if index and middle:
            return Posture.VICTORY
        if index and pinky:
            return Posture.ROCK
        return Posture.NONE
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
    origin: str = "gesture"          # gesture | no-hand | idle | auto-land | disabled
    detail: str = ""                 # що показує кожна рука, напр. "L:POINT R:VICTORY"

    @property
    def action(self) -> int:
        return int(self.vector[4])

    @property
    def action_name(self) -> str:
        return ACTION_NAMES.get(self.action, str(self.action))

    @property
    def gesture_text(self) -> str:
        """Короткий опис жестів обох рук для HUD/логів."""
        return self.detail or str(self.posture)

    def to_payload(self) -> List[float]:
        """JSON-сумісний список для WebSocket (action завжди ціле)."""
        vx, vy, vz, yaw, action = self.vector
        return [round(float(vx), 4), round(float(vy), 4), round(float(vz), 4),
                round(float(yaw), 4), int(action)]

    def to_text(self) -> str:
        vx, vy, vz, yaw, action = self.to_payload()
        return f"[{vx:+.2f}, {vy:+.2f}, {vz:+.2f}, {yaw:+.1f}, {action}]"

    def with_origin(self, origin: str) -> "DroneCommand":
        return DroneCommand(self.vector, self.posture, self.label, origin, self.detail)


HOVER_COMMAND = DroneCommand((0.0, 0.0, 0.0, 0.0, HOVER), Posture.NONE, "HOLD")


# ─────────────────────────────────────────────────────────────────────────────
# Конфігурація та таблиці жестів
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GestureConfig:
    """Налаштування жестів (усі швидкості — у тих самих одиницях, що й вектор)."""

    forward_speed: float = 1.5       # м/с вперед
    backward_speed: float = 1.2      # м/с назад
    strafe_speed: float = 1.2        # м/с вліво / вправо
    yaw_speed: float = 60.0          # °/с поворот вліво / вправо

    image_mirrored: bool = False     # True, якщо кадр дзеркальний (selfie); webcam/ESP32 -> False
    swap_hands: bool = False         # True: права рука = команди, ліва = рух (для лівші)

    stable_time: float = 0.18        # с, скільки жест має протриматись
    hand_lost_timeout: float = 0.35  # с, ігнорування "блимнув" руки
    auto_land_after: float = 8.0     # с без хазяїна -> LAND (0 = вимкнути)

    def __post_init__(self) -> None:
        self.stable_time = max(self.stable_time, 0.0)
        self.hand_lost_timeout = max(self.hand_lost_timeout, 0.0)
        self.auto_land_after = max(self.auto_land_after, 0.0)
        self.forward_speed = max(self.forward_speed, 0.0)
        self.backward_speed = max(self.backward_speed, 0.0)
        self.strafe_speed = max(self.strafe_speed, 0.0)
        self.yaw_speed = max(self.yaw_speed, 0.0)

    def side_for(self, role: str) -> str:
        """Яка рука користувача ("Left"/"Right") виконує роль."""
        command_side = "Right" if self.swap_hands else "Left"
        move_side = "Left" if self.swap_hands else "Right"
        return command_side if role == ROLE_COMMAND else move_side


@dataclass(frozen=True)
class GestureBinding:
    """Що робить дрон, коли розпізнано позу."""

    label: str
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw: float = 0.0            # yaw_rate жесту (°/с), додатне = вправо
    action: int = HOVER

    def as_command(self, posture: Posture) -> DroneCommand:
        return DroneCommand(
            (round(float(self.vx), 4), round(float(self.vy), 4), round(float(self.vz), 4),
             round(float(self.yaw), 4), int(self.action)),
            posture,
            self.label,
        )


Bindings = Dict[Posture, GestureBinding]


def command_bindings(config: Optional[GestureConfig] = None) -> Bindings:
    """Жести руки-КОМАНД (за замовчуванням ліва): зліт, посадка, поворот."""
    cfg = config or GestureConfig()
    land = GestureBinding("LAND", action=LAND)
    return {
        Posture.POINT: GestureBinding("TAKEOFF", action=TAKEOFF),
        Posture.FIST: land,
        Posture.THUMB: land,        # кулак зі стирчачим великим пальцем — теж кулак
        Posture.VICTORY: GestureBinding("TURN LEFT", yaw=-cfg.yaw_speed),
        Posture.PALM: GestureBinding("TURN RIGHT", yaw=cfg.yaw_speed),
    }


def move_bindings(config: Optional[GestureConfig] = None) -> Bindings:
    """Жести руки-РУХУ (за замовчуванням права): вперед, назад, вліво, вправо."""
    cfg = config or GestureConfig()
    left = GestureBinding("LEFT", vy=-cfg.strafe_speed)
    return {
        Posture.POINT: GestureBinding("FORWARD", vx=cfg.forward_speed),
        Posture.VICTORY: GestureBinding("BACKWARD", vx=-cfg.backward_speed),
        Posture.FIST: left,
        Posture.THUMB: left,        # кулак зі стирчачим великим пальцем — теж кулак
        Posture.PALM: GestureBinding("RIGHT", vy=cfg.strafe_speed),
    }


#: Порядок для HUD/README: (роль, поза, як це виглядає)
GESTURE_HELP = (
    (ROLE_COMMAND, Posture.POINT, "☝️ 1 палець"),
    (ROLE_COMMAND, Posture.FIST, "✊ кулак"),
    (ROLE_COMMAND, Posture.VICTORY, "✌️ 2 пальці"),
    (ROLE_COMMAND, Posture.PALM, "✋ долоня"),
    (ROLE_MOVE, Posture.POINT, "☝️ 1 палець"),
    (ROLE_MOVE, Posture.VICTORY, "✌️ 2 пальці"),
    (ROLE_MOVE, Posture.FIST, "✊ кулак"),
    (ROLE_MOVE, Posture.PALM, "✋ долоня"),
)


# ─────────────────────────────────────────────────────────────────────────────
# Стан однієї руки (анти-дриґання)
# ─────────────────────────────────────────────────────────────────────────────
class _HandState:
    """Debounce жесту однієї руки: кандидат -> активний жест після stable_time."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.candidate = Posture.NONE
        self.candidate_since: Optional[float] = None
        self.active = Posture.NONE
        self.last_seen: Optional[float] = None

    def advance(self, posture: Optional[Posture], now: float, config: GestureConfig) -> bool:
        """
        Оновлює стан. posture=None — руки в кадрі немає.
        Повертає True, якщо рука "присутня" (видно зараз або зникла < hand_lost_timeout тому).
        """
        if posture is None:
            if self.last_seen is not None and (now - self.last_seen) < config.hand_lost_timeout:
                return True                      # блимнула — тримаємо попередній жест
            self.reset()
            return False

        self.last_seen = now
        if posture is not self.candidate:
            self.candidate = posture
            self.candidate_since = now
        if self.candidate_since is not None and (now - self.candidate_since) >= config.stable_time:
            self.active = self.candidate
        return True


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
        command_table: Optional[Bindings] = None,
        move_table: Optional[Bindings] = None,
    ) -> None:
        self.config = config or GestureConfig()
        self.command_bindings = command_table if command_table is not None else command_bindings(self.config)
        self.move_bindings = move_table if move_table is not None else move_bindings(self.config)
        self._states: Dict[str, _HandState] = {role: _HandState() for role in ROLES}
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Скидає внутрішній стан (наприклад, після реєстрації нового хазяїна)."""
        for state in self._states.values():
            state.reset()
        self._last_hand_time: Optional[float] = None
        self._last_owner_time: Optional[float] = None
        self.last_command = HOVER_COMMAND

    # ------------------------------------------------------------------ руки
    def user_hand_label(self, label) -> str:
        """
        Мітка руки з погляду КОРИСТУВАЧА.

        MediaPipe вважає, що кадр дзеркальний (selfie). Наші джерела (webcam без
        flip та ESP32) недзеркальні, тому реальна права рука приходить як "Left"
        — міняємо місцями, якщо image_mirrored=False.
        """
        label = str(label or "")
        if self.config.image_mirrored or label not in ("Left", "Right"):
            return label
        return "Right" if label == "Left" else "Left"

    def _side_of_image_half(self, on_image_left: bool) -> str:
        """Яка рука користувача зазвичай з'являється в лівій/правій половині кадру."""
        if self.config.image_mirrored:
            return "Left" if on_image_left else "Right"
        return "Right" if on_image_left else "Left"       # недзеркальний кадр

    def _assign_hands(self, hands, width: float) -> Dict[str, Optional[dict]]:
        """Розкладає руки за ролями: {"command": hand|None, "move": hand|None}."""
        prepared = []
        for hand in hands or ():
            pts = normalize_hand(hand.get("points"))
            if pts is None:
                continue
            prepared.append({"points": pts, "label": self.user_hand_label(hand.get("label"))})

        by_side: Dict[str, dict] = {}
        if len(prepared) >= 2:
            first, second = prepared[0], prepared[1]
            if {first["label"], second["label"]} == {"Left", "Right"}:
                by_side = {first["label"]: first, second["label"]: second}
            else:
                # обом дали однакову мітку — сторону визначає позиція в кадрі
                image_left, image_right = sorted(
                    (first, second), key=lambda h: palm_center(h["points"])[0])
                by_side = {
                    self._side_of_image_half(True): image_left,
                    self._side_of_image_half(False): image_right,
                }
        elif prepared:
            hand = prepared[0]
            side = hand["label"]
            if side not in ("Left", "Right"):
                side = self._side_of_image_half(palm_center(hand["points"])[0] < width / 2.0)
            by_side[side] = hand

        return {role: by_side.get(self.config.side_for(role)) for role in ROLES}

    # ------------------------------------------------------------------ update
    def update(self, hands, frame_shape, now: Optional[float] = None,
               owner_present: Optional[bool] = None) -> DroneCommand:
        """
        Головна функція: руки + розмір кадру + час -> DroneCommand.

        owner_present — чи бачимо ХАЗЯЇНА (не руку!). Від цього залежить авто-посадка:
        хазяїн у кадрі, але рук не видно -> просто hover. Якщо None, вважаємо, що
        хазяїн присутній, коли є хоч одна рука (зворотна сумісність).
        """
        now = now if now is not None else _now()
        cfg = self.config
        width = float(frame_shape[1]) if frame_shape and len(frame_shape) >= 2 else 0.0

        by_role = self._assign_hands(hands, width)
        any_hand = any(hand is not None for hand in by_role.values())
        if owner_present is None:
            owner_present = any_hand
        if owner_present:
            self._last_owner_time = now
        if any_hand:
            self._last_hand_time = now

        present: Dict[str, bool] = {}
        for role in ROLES:
            hand = by_role[role]
            posture = classify_posture(hand["points"]) if hand is not None else None
            present[role] = self._states[role].advance(posture, now, cfg)

        if not any(present.values()):
            return self._no_hand_command(now, owner_present)

        command = self._compose(present)
        self.last_command = command
        return command

    def _compose(self, present: Dict[str, bool]) -> DroneCommand:
        """Склеює жести двох рук в один вектор."""
        tables = {ROLE_COMMAND: self.command_bindings, ROLE_MOVE: self.move_bindings}
        postures: Dict[str, Posture] = {}
        bindings: Dict[str, Optional[GestureBinding]] = {}
        for role in ROLES:
            posture = self._states[role].active if present[role] else Posture.NONE
            postures[role] = posture
            bindings[role] = tables[role].get(posture) if present[role] else None

        vx = vy = vz = yaw = 0.0
        action = HOVER
        labels: List[str] = []
        for role in ROLES:
            binding = bindings[role]
            if binding is None:
                continue
            vx += binding.vx
            vy += binding.vy
            vz += binding.vz
            yaw += binding.yaw
            if action == HOVER and binding.action != HOVER:
                action = binding.action
            labels.append(binding.label)

        if action != HOVER:            # TAKEOFF / LAND мають пріоритет над рухом
            vx = vy = vz = yaw = 0.0
            labels = [ACTION_NAMES.get(action, "ACTION")]

        # "HOVER"/"STOP" показуємо лише коли немає корисніших міток
        useful = [name for name in labels if name not in ("HOVER", "STOP")]
        if useful:
            label = " + ".join(useful)
        else:
            label = labels[0] if labels else "HOLD (no gesture)"

        primary = postures[ROLE_COMMAND] if bindings[ROLE_COMMAND] is not None else postures[ROLE_MOVE]
        return DroneCommand(
            (round(vx, 4), round(vy, 4), round(vz, 4), round(yaw, 4), int(action)),
            primary,
            label,
            "gesture",
            self._detail(present, postures),
        )

    def _detail(self, present: Dict[str, bool], postures: Dict[str, Posture]) -> str:
        """Текст "L:POINT R:VICTORY" — що показує кожна рука (ліва завжди перша)."""
        by_side = {}
        for role in ROLES:
            by_side[self.config.side_for(role)] = str(postures[role]) if present[role] else "-"
        return f"L:{by_side['Left']} R:{by_side['Right']}"

    def _no_hand_command(self, now: float, owner_present: bool) -> DroneCommand:
        """
        Жодної руки (довше за hand_lost_timeout).
          * hover;
          * хазяїна не видно довше auto_land_after — LAND.
        """
        cfg = self.config
        lost_for = None if self._last_hand_time is None else now - self._last_hand_time

        for state in self._states.values():
            state.reset()

        owner_gone_for = None if self._last_owner_time is None else now - self._last_owner_time
        if (cfg.auto_land_after > 0 and not owner_present and owner_gone_for is not None
                and owner_gone_for >= cfg.auto_land_after):
            command = DroneCommand((0.0, 0.0, 0.0, 0.0, LAND), Posture.NONE, "AUTO-LAND", "auto-land")
        else:
            origin = "no-hand" if lost_for is not None or owner_present else "idle"
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


def hand_title(config: GestureConfig, role: str) -> str:
    """Назва руки для таблиць: ЛІВА / ПРАВА."""
    return "ЛІВА" if config.side_for(role) == "Left" else "ПРАВА"


def format_gesture_table(config: Optional[GestureConfig] = None) -> str:
    """Текстова таблиця жестів (для --help та README)."""
    cfg = config or GestureConfig()
    tables = {ROLE_COMMAND: command_bindings(cfg), ROLE_MOVE: move_bindings(cfg)}
    lines = []
    for role, posture, human in GESTURE_HELP:
        binding = tables[role].get(posture)
        vector = binding.as_command(posture).to_text() if binding else "—"
        label = binding.label if binding else "—"
        lines.append(f"  {hand_title(cfg, role):<6} {human:<22} {vector:<34} {label}")
    return "\n".join(lines)
