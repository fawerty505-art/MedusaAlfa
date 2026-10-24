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


class GestureMappingTests(unittest.TestCase):
    """Жест -> вектор [v_x, v_y, v_z, yaw_rate, action]."""

    def setUp(self):
        self.mapper = GestureMapper(GestureConfig(stable_time=0.0))

    def command_for(self, points, label="Right", now=1.0):
        return self.mapper.update([hand_dict(points, label)], FRAME_SHAPE, now=now)

    def test_palm_hovers(self):
        cmd = self.command_for(make_hand(thumb=True))
        self.assertEqual(cmd.posture, Posture.PALM)
        self.assertEqual(cmd.action, HOVER)
        self.assertEqual(cmd.vector[:3], (0.0, 0.0, 0.0))

    def test_fist_lands(self):
        cmd = self.command_for(make_hand(fingers=(0, 0, 0, 0)))
        self.assertEqual(cmd.vector[4], LAND)

    def test_point_takes_off(self):
        cmd = self.command_for(make_hand(fingers=(1, 0, 0, 0)))
        self.assertEqual(cmd.vector[4], TAKEOFF)

    def test_victory_flies_forward(self):
        cmd = self.command_for(make_hand(fingers=(1, 1, 0, 0)))
        self.assertEqual(cmd.vector[0], 1.5)          # approach_speed за замовчуванням
        self.assertEqual(cmd.action, HOVER)

    def test_three_retreats(self):
        cmd = self.command_for(make_hand(fingers=(1, 1, 1, 0)))
        self.assertEqual(cmd.vector[0], -1.2)         # retreat_speed за замовчуванням

    def test_rock_descends(self):
        cmd = self.command_for(make_hand(fingers=(1, 0, 0, 1)))
        self.assertEqual(cmd.vector[2], -0.9)

    def test_relaxed_open_hand_never_descends(self):
        """Регресія: відкрита рука з притиснутим великим пальцем раніше читалась як 'вниз'."""
        for thumb in (False, True):
            with self.subTest(thumb=thumb):
                cmd = self.command_for(make_hand(fingers=(1, 1, 1, 1), thumb=thumb))
                self.assertEqual(cmd.vector[:3], (0.0, 0.0, 0.0))
                self.assertEqual(cmd.action, HOVER)

    def test_thumb_climbs(self):
        cmd = self.command_for(make_hand(fingers=(0, 0, 0, 0), thumb=True))
        self.assertEqual(cmd.vector[2], 0.9)

    def test_custom_speeds_are_used(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, approach_speed=3.0,
                                             climb_speed=2.0))
        cmd = mapper.update([hand_dict(make_hand(fingers=(1, 1, 0, 0)))], FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.vector[0], 3.0)
        cmd = mapper.update([hand_dict(make_hand(fingers=(0, 0, 0, 0), thumb=True))],
                            FRAME_SHAPE, now=2.0)
        self.assertEqual(cmd.vector[2], 2.0)

    def test_payload_format(self):
        cmd = self.command_for(make_hand(fingers=(1, 1, 0, 0)))
        payload = cmd.to_payload()
        self.assertEqual(len(payload), 5)
        self.assertIsInstance(payload[4], int)
        self.assertTrue(all(isinstance(v, (int, float)) for v in payload))
        self.assertEqual(cmd.to_text().count(","), 4)


class GestureStabilityTests(unittest.TestCase):
    def test_gesture_needs_to_be_stable(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.20))
        hand = [hand_dict(make_hand(fingers=(1, 1, 0, 0)))]

        first = mapper.update(hand, FRAME_SHAPE, now=10.0)
        self.assertEqual(first.label, "HOLD (no gesture)")

        still = mapper.update(hand, FRAME_SHAPE, now=10.10)
        self.assertEqual(still.label, "HOLD (no gesture)")

        settled = mapper.update(hand, FRAME_SHAPE, now=10.21)
        self.assertEqual(settled.label, "FORWARD / APPROACH")
        self.assertEqual(settled.vector[0], 1.5)

    def test_previous_gesture_kept_while_candidate_not_stable(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.20))
        pose_a = [hand_dict(make_hand(fingers=(1, 1, 1, 1)))]     # PALM -> hover
        pose_b = [hand_dict(make_hand(fingers=(0, 0, 0, 0)))]     # FIST -> land

        mapper.update(pose_a, FRAME_SHAPE, now=0.0)
        active = mapper.update(pose_a, FRAME_SHAPE, now=0.25)
        self.assertEqual(active.posture, Posture.PALM)

        flicker = mapper.update(pose_b, FRAME_SHAPE, now=0.30)
        self.assertEqual(flicker.posture, Posture.PALM)           # ще не стабільно
        settled = mapper.update(pose_b, FRAME_SHAPE, now=0.55)
        self.assertEqual(settled.vector[4], LAND)



class HandLossTests(unittest.TestCase):
    def test_no_hand_sends_hover(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0))
        cmd = mapper.update([], FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.vector, (0.0, 0.0, 0.0, 0.0, HOVER))
        self.assertEqual(cmd.origin, "idle")

    def test_auto_land_after_owner_disappears(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, auto_land_after=2.0))
        mapper.update([hand_dict(make_hand())], FRAME_SHAPE, now=0.0)

        short_gap = mapper.update([], FRAME_SHAPE, now=1.0)
        self.assertEqual(short_gap.vector[4], HOVER)
        self.assertEqual(short_gap.origin, "no-hand")

        landed = mapper.update([], FRAME_SHAPE, now=2.5)
        self.assertEqual(landed.vector[4], LAND)
        self.assertEqual(landed.origin, "auto-land")

        # хазяїн повернувся -> авто-лендінг скасовується
        back = mapper.update([hand_dict(make_hand())], FRAME_SHAPE, now=3.0)
        self.assertEqual(back.vector[4], HOVER)

    def test_owner_visible_without_hand_never_auto_lands(self):
        """Регресія: хазяїн стоїть у кадрі з опущеними руками -> hover, а не посадка."""
        mapper = GestureMapper(GestureConfig(stable_time=0.0, auto_land_after=2.0))
        mapper.update([hand_dict(make_hand())], FRAME_SHAPE, now=0.0, owner_present=True)
        cmd = mapper.update([], FRAME_SHAPE, now=60.0, owner_present=True)
        self.assertEqual(cmd.vector[4], HOVER)
        self.assertEqual(cmd.origin, "no-hand")

    def test_owner_gone_triggers_auto_land_even_without_prior_hand(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, auto_land_after=2.0))
        mapper.update([], FRAME_SHAPE, now=0.0, owner_present=True)
        cmd = mapper.update([], FRAME_SHAPE, now=3.0, owner_present=False)
        self.assertEqual(cmd.vector[4], LAND)

    def test_short_hand_dropout_keeps_gesture(self):
        """hand_lost_timeout: один кадр без детекції не перериває маневр."""
        mapper = GestureMapper(GestureConfig(stable_time=0.0, hand_lost_timeout=0.35))
        fwd = mapper.update([hand_dict(make_hand(fingers=(1, 1, 0, 0)))], FRAME_SHAPE, now=0.0,
                            owner_present=True)
        blink = mapper.update([], FRAME_SHAPE, now=0.10, owner_present=True)
        self.assertEqual(blink.vector, fwd.vector)
        gone = mapper.update([], FRAME_SHAPE, now=0.60, owner_present=True)
        self.assertEqual(gone.vector, (0.0, 0.0, 0.0, 0.0, HOVER))

    def test_auto_land_can_be_disabled(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, auto_land_after=0.0))
        mapper.update([hand_dict(make_hand())], FRAME_SHAPE, now=0.0)
        cmd = mapper.update([], FRAME_SHAPE, now=100.0)
        self.assertEqual(cmd.vector[4], HOVER)


class YawFromHandTests(unittest.TestCase):
    def test_user_right_turns_right_on_non_mirrored_frame(self):
        """Недзеркальна камера: рука праворуч від КОРИСТУВАЧА видна ЛІВОРУЧ у кадрі."""
        mapper = GestureMapper(GestureConfig(stable_time=0.0))
        cmd = mapper.update([hand_dict(make_hand(cx=CENTER_X - 200))], FRAME_SHAPE, now=1.0)
        self.assertGreater(cmd.vector[3], 5.0)       # yaw_rate > 0 = вправо

    def test_user_left_turns_left_on_non_mirrored_frame(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0))
        cmd = mapper.update([hand_dict(make_hand(cx=CENTER_X + 200))], FRAME_SHAPE, now=1.0)
        self.assertLess(cmd.vector[3], -5.0)

    def test_mirrored_frame_uses_image_side_directly(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, image_mirrored=True))
        cmd = mapper.update([hand_dict(make_hand(cx=CENTER_X + 200))], FRAME_SHAPE, now=1.0)
        self.assertGreater(cmd.vector[3], 5.0)

    def test_hand_in_the_centre_does_not_turn(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0))
        cmd = mapper.update([hand_dict(make_hand(cx=CENTER_X - 4))], FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.vector[3], 0.0)

    def test_yaw_disabled(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, yaw_from_hand=False))
        cmd = mapper.update([hand_dict(make_hand(cx=CENTER_X + 250))], FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.vector[3], 0.0)

    def test_action_gestures_never_yaw(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0))
        cmd = mapper.update([hand_dict(make_hand(fingers=(0, 0, 0, 0), cx=CENTER_X + 250))],
                            FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.vector[4], LAND)
        self.assertEqual(cmd.vector[3], 0.0)

    def test_yaw_invert(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, yaw_invert=True))
        cmd = mapper.update([hand_dict(make_hand(cx=CENTER_X + 200))], FRAME_SHAPE, now=1.0)
        self.assertGreater(cmd.vector[3], 5.0)


class MultiHandTests(unittest.TestCase):
    """MediaPipe на недзеркальному кадрі інвертує мітки: його 'Left' = права рука користувача."""

    def test_prefers_users_right_hand_on_non_mirrored_frame(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, prefer_hand="Right"))
        hands = [
            hand_dict(make_hand(fingers=(1, 1, 0, 0)), label="Right"),   # = ліва користувача, V
            hand_dict(make_hand(), label="Left"),                        # = права користувача, PALM
        ]
        cmd = mapper.update(hands, FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.posture, Posture.PALM)

    def test_mirrored_frame_keeps_mediapipe_labels(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, prefer_hand="Right", image_mirrored=True))
        hands = [
            hand_dict(make_hand(fingers=(1, 1, 0, 0)), label="Left"),
            hand_dict(make_hand(), label="Right"),
        ]
        cmd = mapper.update(hands, FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.posture, Posture.PALM)

    def test_falls_back_to_any_hand(self):
        mapper = GestureMapper(GestureConfig(stable_time=0.0, prefer_hand="Right"))
        hands = [hand_dict(make_hand(fingers=(1, 0, 0, 0)), label="Right")]   # лише ліва користувача
        cmd = mapper.update(hands, FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.vector[4], TAKEOFF)


class CommandTests(unittest.TestCase):
    def test_hover_command_shape(self):
        cmd = DroneCommand((0.0, 0.0, 0.0, 0.0, HOVER))
        self.assertEqual(cmd.to_payload(), [0.0, 0.0, 0.0, 0.0, 0])
        self.assertEqual(cmd.action_name, "HOVER")
        self.assertEqual(cmd.origin, "gesture")

    def test_with_origin_is_a_copy(self):
        cmd = DroneCommand((1.0, 0.0, 0.0, 0.0, HOVER), Posture.VICTORY, "FORWARD")
        disabled = cmd.with_origin("disabled")
        self.assertEqual(disabled.origin, "disabled")
        self.assertEqual(cmd.origin, "gesture")
        self.assertEqual(disabled.vector, cmd.vector)


if __name__ == "__main__":
    unittest.main()

