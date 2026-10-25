"""Тести розпізнавання жестів та мапінгу їх у вектор команд дрона.

Кисть синтезується геометрично (21 точка, як у MediaPipe), тому тести не
потребують ні камери, ні MediaPipe/YOLO.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medusa.gestures import (  # noqa: E402
    HOVER,
    LAND,
    TAKEOFF,
    DroneCommand,
    GestureConfig,
    GestureMapper,
    Posture,
    classify_posture,
    count_fingers,
    fingers_up,
    format_gesture_table,
    palm_center,
)

FRAME_SHAPE = (480, 640, 3)         # 640x480 VGA, як у MedusaEye
CENTER_X = FRAME_SHAPE[1] / 2


def make_hand(fingers=(1, 1, 1, 1, 1), thumb=False, cx=CENTER_X, cy=300.0, scale=1.0):
    """
    Синтезує 21 точку правої кисті у координатах кадру.

    fingers — (index, middle, ring, pinky), 1 = розігнутий, 0 = зігнутий.
    Пропорції взяті з реальної долоні: зап'ястя нижче, кінчики пальців вище.
    """
    ox, oy = float(cx), float(cy)

    def p(x, y):
        return (ox + x * scale, oy + y * scale)

    points = [None] * 21
    points[0] = p(0, 40)                                  # wrist

    # великий палець
    points[1] = p(-20, 30)
    points[2] = p(-32, 20)
    points[3] = p(-42, 8)
    points[4] = p(-52, -6) if thumb else p(-12, 22)        # TIP

    # (mcp, pip, dip, tip_extended, tip_folded)
    layout = [
        (5, p(-18, 5), p(-18, -25), p(-18, -42), p(-18, -55), p(-18, 18)),
        (9, p(-3, 0), p(-3, -30), p(-3, -48), p(-3, -62), p(-3, 20)),
        (13, p(12, 5), p(12, -26), p(12, -46), p(12, -56), p(12, 16)),
        (17, p(25, 12), p(25, -15), p(25, -32), p(25, -42), p(25, 24)),
    ]
    for (mcp, mcp_pt, pip_pt, dip_pt, tip_open, tip_closed), extended in zip(layout, fingers):
        points[mcp] = mcp_pt
        points[mcp + 1] = pip_pt
        points[mcp + 2] = dip_pt
        points[mcp + 3] = tip_open if extended else tip_closed
    return points


def hand_dict(points, label="Right"):
    return {"points": points, "wrist": points[0], "label": label}


class FingerDetectionTests(unittest.TestCase):
    def test_open_palm_has_five_fingers_up(self):
        self.assertEqual(count_fingers(make_hand(thumb=True)), 5)

    def test_fist_has_no_fingers_up(self):
        self.assertEqual(count_fingers(make_hand(fingers=(0, 0, 0, 0), thumb=False)), 0)

    def test_thumb_flag_recognised(self):
        thumb, index, middle, ring, pinky = fingers_up(
            make_hand(fingers=(0, 0, 0, 0), thumb=True))
        self.assertTrue(thumb)
        self.assertFalse(index or middle or ring or pinky)

    def test_pointing_hand(self):
        self.assertEqual(count_fingers(make_hand(fingers=(1, 0, 0, 0))), 1)

    def test_scaled_and_moved_hand_is_still_recognised(self):
        palm = make_hand(thumb=True, cx=120, cy=430, scale=0.55)
        self.assertEqual(classify_posture(palm), Posture.PALM)

    def test_too_few_points_returns_none_posture(self):
        self.assertEqual(classify_posture([(0, 0)] * 5), Posture.NONE)
        self.assertEqual(classify_posture(None), Posture.NONE)


class PostureClassificationTests(unittest.TestCase):
    CASES = [
        (Posture.PALM, dict(fingers=(1, 1, 1, 1), thumb=True)),
        (Posture.PALM, dict(fingers=(1, 1, 1, 1), thumb=False)),   # великий палець не важливий
        (Posture.FIST, dict(fingers=(0, 0, 0, 0), thumb=False)),
        (Posture.THUMB, dict(fingers=(0, 0, 0, 0), thumb=True)),
        (Posture.POINT, dict(fingers=(1, 0, 0, 0))),
        (Posture.VICTORY, dict(fingers=(1, 1, 0, 0))),
        (Posture.THREE, dict(fingers=(1, 1, 1, 0))),
        (Posture.ROCK, dict(fingers=(1, 0, 0, 1))),
    ]

    def test_postures(self):
        for expected, kwargs in self.CASES:
            with self.subTest(posture=expected.value):
                self.assertEqual(classify_posture(make_hand(**kwargs)), expected)

    def test_ring_only_is_not_a_command(self):
        self.assertEqual(classify_posture(make_hand(fingers=(0, 0, 1, 0))), Posture.NONE)

    def test_palm_center_is_between_mcp_points(self):
        cx, cy = palm_center(make_hand())
        self.assertAlmostEqual(cx, CENTER_X + 4.0, delta=1.0)
        self.assertAlmostEqual(cy, 300 + 5.5, delta=2.0)


def user_left(points):
    """Ліва рука КОРИСТУВАЧА на недзеркальному кадрі: MediaPipe називає її 'Right'."""
    return hand_dict(points, "Right")


def user_right(points):
    """Права рука КОРИСТУВАЧА на недзеркальному кадрі: MediaPipe називає її 'Left'."""
    return hand_dict(points, "Left")


POINT = dict(fingers=(1, 0, 0, 0))
VICTORY = dict(fingers=(1, 1, 0, 0))
THREE = dict(fingers=(1, 1, 1, 0))
PALM = dict(fingers=(1, 1, 1, 1))
FIST = dict(fingers=(0, 0, 0, 0))


class CommandHandTests(unittest.TestCase):
    """ЛІВА рука = команди: зліт / посадка / поворот."""

    def setUp(self):
        self.mapper = GestureMapper(GestureConfig(stable_time=0.0))

    def cmd(self, **hand):
        return self.mapper.update([user_left(make_hand(**hand))], FRAME_SHAPE, now=1.0)

    def test_one_finger_takes_off(self):
        cmd = self.cmd(**POINT)
        self.assertEqual(cmd.vector, (0.0, 0.0, 0.0, 0.0, TAKEOFF))

    def test_fist_lands(self):
        self.assertEqual(self.cmd(**FIST).vector[4], LAND)

    def test_fist_with_thumb_out_still_lands(self):
        self.assertEqual(self.cmd(fingers=(0, 0, 0, 0), thumb=True).vector[4], LAND)

    def test_two_fingers_turn_left(self):
        cmd = self.cmd(**VICTORY)
        self.assertEqual(cmd.vector, (0.0, 0.0, 0.0, -60.0, HOVER))

    def test_palm_turns_right(self):
        cmd = self.cmd(**PALM)
        self.assertEqual(cmd.vector, (0.0, 0.0, 0.0, 60.0, HOVER))
        self.assertEqual(cmd.label, "TURN RIGHT")

    def test_palm_ignores_thumb(self):
        for thumb in (False, True):
            with self.subTest(thumb=thumb):
                self.assertEqual(self.cmd(fingers=(1, 1, 1, 1), thumb=thumb).vector[3], 60.0)

    def test_three_fingers_do_nothing(self):
        """3 пальці більше не жест: рука нічого не додає до команди."""
        self.assertEqual(self.cmd(**THREE).vector, (0.0, 0.0, 0.0, 0.0, HOVER))

    def test_hand_position_in_frame_does_not_change_turn(self):
        """Поворот залежить лише від жесту, а не від зсуву руки в кадрі."""
        for cx in (80, CENTER_X, 560):
            with self.subTest(cx=cx):
                mapper = GestureMapper(GestureConfig(stable_time=0.0))
                cmd = mapper.update([user_left(make_hand(cx=cx, **PALM))], FRAME_SHAPE, now=1.0)
                self.assertEqual(cmd.vector[3], 60.0)


class MoveHandTests(unittest.TestCase):
    """ПРАВА рука = рух: вперед / назад / вліво / вправо."""

    def setUp(self):
        self.mapper = GestureMapper(GestureConfig(stable_time=0.0))

    def cmd(self, **hand):
        return self.mapper.update([user_right(make_hand(**hand))], FRAME_SHAPE, now=1.0)

    def test_one_finger_flies_forward(self):
        self.assertEqual(self.cmd(**POINT).vector, (1.5, 0.0, 0.0, 0.0, HOVER))

    def test_two_fingers_fly_backward(self):
        self.assertEqual(self.cmd(**VICTORY).vector, (-1.2, 0.0, 0.0, 0.0, HOVER))

    def test_fist_strafes_left(self):
        self.assertEqual(self.cmd(**FIST).vector, (0.0, -1.2, 0.0, 0.0, HOVER))
        self.assertEqual(self.cmd(fingers=(0, 0, 0, 0), thumb=True).vector[1], -1.2)

    def test_three_fingers_do_nothing(self):
        self.assertEqual(self.cmd(**THREE).vector, (0.0, 0.0, 0.0, 0.0, HOVER))

    def test_palm_strafes_right(self):
        self.assertEqual(self.cmd(**PALM).vector, (0.0, 1.2, 0.0, 0.0, HOVER))

    def test_palm_ignores_thumb(self):
        for thumb in (False, True):
            with self.subTest(thumb=thumb):
                self.assertEqual(self.cmd(fingers=(1, 1, 1, 1), thumb=thumb).vector[1], 1.2)

    def test_move_hand_cannot_take_off_or_land(self):
        for hand in (POINT, FIST, PALM):
            self.assertEqual(self.cmd(**hand).action, HOVER)

    def test_custom_speeds_are_used(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, forward_speed=3.0, strafe_speed=0.5))
        cmd = mapper.update([user_right(make_hand(**POINT))], FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.vector[0], 3.0)
        cmd = mapper.update([user_right(make_hand(**PALM))], FRAME_SHAPE, now=2.0)
        self.assertEqual(cmd.vector[1], 0.5)


class TwoHandCombinationTests(unittest.TestCase):
    def setUp(self):
        self.mapper = GestureMapper(GestureConfig(stable_time=0.0))

    def two(self, left, right, now=1.0):
        hands = [user_left(make_hand(cx=490, **left)), user_right(make_hand(cx=150, **right))]
        return self.mapper.update(hands, FRAME_SHAPE, now=now)

    def test_forward_while_turning_right(self):
        cmd = self.two(left=PALM, right=POINT)
        self.assertEqual(cmd.vector, (1.5, 0.0, 0.0, 60.0, HOVER))
        self.assertIn("FORWARD", cmd.label)
        self.assertIn("TURN RIGHT", cmd.label)

    def test_strafe_left_while_turning_left(self):
        cmd = self.two(left=VICTORY, right=FIST)
        self.assertEqual(cmd.vector, (0.0, -1.2, 0.0, -60.0, HOVER))

    def test_strafe_right_while_turning_right(self):
        cmd = self.two(left=PALM, right=PALM)
        self.assertEqual(cmd.vector, (0.0, 1.2, 0.0, 60.0, HOVER))

    def test_backward_while_turning_left(self):
        cmd = self.two(left=VICTORY, right=VICTORY)
        self.assertEqual(cmd.vector, (-1.2, 0.0, 0.0, -60.0, HOVER))

    def test_takeoff_overrides_movement(self):
        cmd = self.two(left=POINT, right=POINT)
        self.assertEqual(cmd.vector, (0.0, 0.0, 0.0, 0.0, TAKEOFF))

    def test_land_overrides_movement(self):
        cmd = self.two(left=FIST, right=PALM)
        self.assertEqual(cmd.vector, (0.0, 0.0, 0.0, 0.0, LAND))

    def test_order_of_hands_in_list_does_not_matter(self):
        hands = [user_right(make_hand(cx=150, **POINT)), user_left(make_hand(cx=490, **PALM))]
        cmd = self.mapper.update(hands, FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.vector, (1.5, 0.0, 0.0, 60.0, HOVER))

    def test_detail_shows_both_hands(self):
        cmd = self.two(left=POINT, right=VICTORY)
        self.assertEqual(cmd.detail, "L:POINT R:VICTORY")
        self.assertEqual(cmd.gesture_text, "L:POINT R:VICTORY")


class HandAssignmentTests(unittest.TestCase):
    """Яка рука ліва/права: мітки MediaPipe, позиція в кадрі, mirrored, swap."""

    def test_mirrored_frame_keeps_mediapipe_labels(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, image_mirrored=True))
        cmd = mapper.update([hand_dict(make_hand(**POINT), "Left")], FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.vector[4], TAKEOFF)           # 'Left' на дзеркальному кадрі = ліва

    def test_swap_hands_exchanges_roles(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, swap_hands=True))
        cmd = mapper.update([user_right(make_hand(**POINT))], FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.vector[4], TAKEOFF)           # права = команди
        cmd = mapper.update([user_left(make_hand(**POINT))], FRAME_SHAPE, now=2.0)
        self.assertEqual(cmd.vector, (1.5, 0.0, 0.0, 0.0, HOVER))   # ліва = рух

    def test_same_label_on_both_hands_uses_position(self):
        """MediaPipe іноді дає обом рукам однакову мітку: тоді рішає позиція в кадрі."""
        mapper = GestureMapper(GestureConfig(stable_time=0.0))
        # недзеркальний кадр: ліва рука користувача — праворуч у кадрі
        hands = [
            hand_dict(make_hand(cx=150, **POINT), "Left"),      # зліва в кадрі = ПРАВА користувача -> вперед
            hand_dict(make_hand(cx=490, **PALM), "Left"),       # справа в кадрі = ЛІВА користувача -> поворот вправо
        ]
        cmd = mapper.update(hands, FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.vector, (1.5, 0.0, 0.0, 60.0, HOVER))

    def test_unlabelled_single_hand_uses_position(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0))
        cmd = mapper.update([hand_dict(make_hand(cx=500, **POINT), "Hand")], FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.vector[4], TAKEOFF)           # справа в кадрі = ліва користувача

    def test_unlabelled_hand_mirrored_frame(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, image_mirrored=True))
        cmd = mapper.update([hand_dict(make_hand(cx=100, **POINT), "Hand")], FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.vector[4], TAKEOFF)           # зліва в кадрі = ліва користувача

    def test_hand_labels_helper(self):
        mapper = GestureMapper(GestureConfig())
        self.assertEqual(mapper.user_hand_label("Left"), "Right")
        self.assertEqual(mapper.user_hand_label("Right"), "Left")
        self.assertEqual(mapper.user_hand_label("Hand"), "Hand")


class DebounceTests(unittest.TestCase):
    def test_gesture_must_be_stable(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.2))
        hands = [user_left(make_hand(**POINT))]
        self.assertEqual(mapper.update(hands, FRAME_SHAPE, now=1.00).vector[4], HOVER)
        self.assertEqual(mapper.update(hands, FRAME_SHAPE, now=1.10).vector[4], HOVER)
        self.assertEqual(mapper.update(hands, FRAME_SHAPE, now=1.25).vector[4], TAKEOFF)

    def test_each_hand_is_debounced_independently(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.2))
        left = user_left(make_hand(cx=490, **POINT))
        right = user_right(make_hand(cx=150, **POINT))
        mapper.update([right], FRAME_SHAPE, now=1.0)
        mapper.update([right], FRAME_SHAPE, now=1.3)               # права вже стабільна
        cmd = mapper.update([left, right], FRAME_SHAPE, now=1.35)  # ліва щойно з'явилась
        self.assertEqual(cmd.vector, (1.5, 0.0, 0.0, 0.0, HOVER))  # TAKEOFF ще не активний
        cmd = mapper.update([left, right], FRAME_SHAPE, now=1.6)
        self.assertEqual(cmd.vector[4], TAKEOFF)

    def test_flicker_does_not_change_command(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.2))
        hands = [user_right(make_hand(**POINT))]
        mapper.update(hands, FRAME_SHAPE, now=1.0)
        mapper.update(hands, FRAME_SHAPE, now=1.3)
        mapper.update([user_right(make_hand(**PALM))], FRAME_SHAPE, now=1.35)   # 1 кадр "не те"
        cmd = mapper.update(hands, FRAME_SHAPE, now=1.40)
        self.assertEqual(cmd.vector[0], 1.5)


class HandLossTests(unittest.TestCase):
    def test_short_dropout_keeps_command(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, hand_lost_timeout=0.35))
        mapper.update([user_right(make_hand(**POINT))], FRAME_SHAPE, now=1.0)
        cmd = mapper.update([], FRAME_SHAPE, now=1.2, owner_present=True)
        self.assertEqual(cmd.vector[0], 1.5)

    def test_long_dropout_hovers(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, hand_lost_timeout=0.35))
        mapper.update([user_right(make_hand(**POINT))], FRAME_SHAPE, now=1.0)
        cmd = mapper.update([], FRAME_SHAPE, now=1.6, owner_present=True)
        self.assertEqual(cmd.vector, (0.0, 0.0, 0.0, 0.0, HOVER))

    def test_one_hand_lost_other_keeps_working(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, hand_lost_timeout=0.35))
        both = [user_left(make_hand(cx=490, **PALM)), user_right(make_hand(cx=150, **POINT))]
        mapper.update(both, FRAME_SHAPE, now=1.0)
        cmd = mapper.update([user_right(make_hand(cx=150, **POINT))], FRAME_SHAPE, now=1.8)
        self.assertEqual(cmd.vector, (1.5, 0.0, 0.0, 0.0, HOVER))     # поворот зник, рух лишився

    def test_owner_visible_without_hands_does_not_land(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, auto_land_after=1.0))
        mapper.update([user_right(make_hand(**PALM))], FRAME_SHAPE, now=0.0)
        cmd = mapper.update([], FRAME_SHAPE, now=30.0, owner_present=True)
        self.assertEqual(cmd.vector[4], HOVER)

    def test_owner_lost_lands(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, auto_land_after=1.0))
        mapper.update([user_right(make_hand(**PALM))], FRAME_SHAPE, now=0.0)
        cmd = mapper.update([], FRAME_SHAPE, now=5.0, owner_present=False)
        self.assertEqual(cmd.vector[4], LAND)
        self.assertEqual(cmd.origin, "auto-land")

    def test_auto_land_can_be_disabled(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, auto_land_after=0))
        mapper.update([user_right(make_hand(**PALM))], FRAME_SHAPE, now=0.0)
        cmd = mapper.update([], FRAME_SHAPE, now=500.0, owner_present=False)
        self.assertEqual(cmd.vector[4], HOVER)


class CommandTests(unittest.TestCase):
    def test_hover_command_shape(self):
        cmd = DroneCommand((0.0, 0.0, 0.0, 0.0, HOVER))
        self.assertEqual(cmd.to_payload(), [0.0, 0.0, 0.0, 0.0, 0])
        self.assertEqual(cmd.action_name, "HOVER")
        self.assertEqual(cmd.origin, "gesture")

    def test_payload_format(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0))
        cmd = mapper.update([user_right(make_hand(**POINT))], FRAME_SHAPE, now=1.0)
        payload = cmd.to_payload()
        self.assertEqual(len(payload), 5)
        self.assertIsInstance(payload[4], int)
        self.assertTrue(all(isinstance(v, (int, float)) for v in payload))

    def test_with_origin_is_a_copy(self):
        cmd = DroneCommand((1.0, 0.0, 0.0, 0.0, HOVER), Posture.POINT, "FORWARD", detail="L:- R:POINT")
        disabled = cmd.with_origin("disabled")
        self.assertEqual(disabled.origin, "disabled")
        self.assertEqual(cmd.origin, "gesture")
        self.assertEqual(disabled.vector, cmd.vector)
        self.assertEqual(disabled.detail, cmd.detail)


class GestureTableTests(unittest.TestCase):
    def test_table_lists_both_hands(self):
        table = format_gesture_table()
        self.assertIn("ЛІВА", table)
        self.assertIn("ПРАВА", table)
        for label in ("TAKEOFF", "LAND", "TURN LEFT", "TURN RIGHT",
                      "FORWARD", "BACKWARD", "LEFT", "RIGHT"):
            self.assertIn(label, table)
        self.assertEqual(len(table.splitlines()), 8)       # 4 жести на кожну руку

    def test_table_follows_swap_hands(self):
        first_line = format_gesture_table(GestureConfig(swap_hands=True)).splitlines()[0]
        self.assertIn("ПРАВА", first_line)
        self.assertIn("TAKEOFF", first_line)


if __name__ == "__main__":
    unittest.main()
