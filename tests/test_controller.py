"""Тести GestureDroneController (без мережі та без камери)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medusa.controller import GestureDroneController  # noqa: E402
from medusa.gestures import HOVER, LAND, GestureConfig, GestureMapper, Posture  # noqa: E402
from tests.test_gestures import FRAME_SHAPE, hand_dict, make_hand  # noqa: E402


class StubLink:
    """Підміна DroneLink: просто запам'ятовує команди."""

    url = "stub://simulator"
    connected = True

    def __init__(self):
        self.sent = []
        self.stopped = False

    def send(self, command):
        self.sent.append(command)
        return command.to_payload()

    def status_text(self):
        return "CONNECTED"

    def telemetry(self):
        return None

    def stop(self):
        self.stopped = True


def assoc(points, label="Right"):
    """Імітує словник, який повертає MedusaEye.process_pipeline()."""
    return {
        "owner_face": None,
        "owner_body": {"box": (10, 10, 200, 300), "track_id": 1},
        "owner_hands": [hand_dict(points, label)],
        "others": [],
    }


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.link = StubLink()
        self.controller = GestureDroneController(
            self.link, GestureMapper(GestureConfig(stable_time=0.0)))

    def test_sends_command_for_recognised_gesture(self):
        cmd = self.controller.update(assoc(make_hand(fingers=(1, 1, 0, 0))), FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.posture, Posture.VICTORY)
        self.assertEqual(len(self.link.sent), 1)
        self.assertEqual(self.link.sent[0].vector[0], 1.5)

    def test_accepts_hands_list_directly(self):
        cmd = self.controller.update([hand_dict(make_hand(fingers=(0, 0, 0, 0)))],
                                     FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.action, LAND)

    def test_disabled_controller_does_not_send(self):
        self.controller.enabled = False
        cmd = self.controller.update(assoc(make_hand()), FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.origin, "disabled")
        self.assertEqual(self.link.sent, [])

    def test_toggle_switches_mode(self):
        self.assertTrue(self.controller.enabled)
        self.assertFalse(self.controller.toggle())
        self.assertTrue(self.controller.toggle())

    def test_toggle_off_sends_explicit_hover(self):
        self.controller.update(assoc(make_hand(fingers=(1, 1, 0, 0))), FRAME_SHAPE, now=1.0)
        self.controller.toggle()                       # OFF
        self.assertEqual(self.link.sent[-1].vector, (0.0, 0.0, 0.0, 0.0, HOVER))

    def test_owner_visible_without_hands_does_not_land(self):
        controller = GestureDroneController(
            self.link, GestureMapper(GestureConfig(stable_time=0.0, auto_land_after=1.0)))
        controller.update(assoc(make_hand()), FRAME_SHAPE, now=0.0)
        no_hands = {"owner_body": {"box": (0, 0, 10, 10)}, "owner_hands": []}
        cmd = controller.update(no_hands, FRAME_SHAPE, now=30.0)
        self.assertEqual(cmd.vector[4], HOVER)

    def test_owner_lost_lands(self):
        controller = GestureDroneController(
            self.link, GestureMapper(GestureConfig(stable_time=0.0, auto_land_after=1.0)))
        controller.update(assoc(make_hand()), FRAME_SHAPE, now=0.0)
        cmd = controller.update({"owner_body": None, "owner_hands": []}, FRAME_SHAPE, now=5.0)
        self.assertEqual(cmd.vector[4], LAND)

    def test_toggle_off_resets_mapper_state(self):
        self.controller.update(assoc(make_hand(fingers=(1, 1, 1, 1))), FRAME_SHAPE, now=1.0)
        self.controller.toggle()                       # OFF
        self.assertEqual(self.controller.mapper._active, Posture.NONE)

    def test_no_hand_sends_hover(self):
        cmd = self.controller.update({"owner_hands": []}, FRAME_SHAPE, now=1.0)
        self.assertEqual(cmd.vector[4], HOVER)
        self.assertEqual(len(self.link.sent), 1)

    def test_close_stops_link(self):
        self.controller.close()
        self.assertTrue(self.link.stopped)


if __name__ == "__main__":
    unittest.main()
