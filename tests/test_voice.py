"""Тести голосових команд (ElevenLabs STT) — без мікрофона, без мережі.

HTTP-запит до ElevenLabs підміняється через unittest.mock, тож тести не
витрачають квоту і працюють офлайн. Живий запит перевірено окремо (див. README).
"""

import io
import os
import sys
import tempfile
import unittest
import wave
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

from medusa import voice  # noqa: E402
from medusa.controller import VOICE_HOLD_SECONDS, GestureDroneController  # noqa: E402
from medusa.gestures import HOVER, LAND, TAKEOFF, GestureConfig, GestureMapper  # noqa: E402
from medusa.voice import VoiceCommander, VoiceError, parse_voice_command  # noqa: E402
from tests.test_gestures import FRAME_SHAPE, make_hand, user_left  # noqa: E402
from tests.test_controller import StubLink  # noqa: E402


class ParseVoiceCommandTests(unittest.TestCase):
    def test_fly_maps_to_word(self):
        self.assertEqual(parse_voice_command("fly"), "fly")
        self.assertEqual(parse_voice_command("Fly."), "fly")
        self.assertEqual(parse_voice_command("  SIT!  "), "sit")

    def test_unknown_or_ambiguous_is_none(self):
        self.assertIsNone(parse_voice_command("hello there"))
        self.assertIsNone(parse_voice_command(""))
        self.assertIsNone(parse_voice_command(None))
        self.assertIsNone(parse_voice_command("fly sit"))     # неоднозначно — не виконуємо

    def test_does_not_match_inside_other_words(self):
        self.assertIsNone(parse_voice_command("flyer"))
        self.assertIsNone(parse_voice_command("sitting"))

    def test_actions_match_simulator_protocol(self):
        self.assertEqual(voice.VOICE_COMMANDS["fly"], TAKEOFF)
        self.assertEqual(voice.VOICE_COMMANDS["sit"], LAND)


class WavTests(unittest.TestCase):
    def test_make_wav_is_16khz_mono_pcm16(self):
        pcm = b"\x01\x00" * 1600          # 0.1 с
        with wave.open(io.BytesIO(voice.make_wav(pcm)), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), 16000)
            self.assertEqual(wav_file.readframes(10000), pcm)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class TranscribeTests(unittest.TestCase):
    def test_posts_to_elevenlabs_with_key_header_and_model(self):
        response = FakeResponse(payload={"text": "Fly", "language_code": "eng"})
        with mock.patch.object(voice.requests, "post", return_value=response) as post:
            text = voice.transcribe(b"WAVDATA", "sk_test", "en")
        self.assertEqual(text, "Fly")
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.elevenlabs.io/v1/speech-to-text")
        self.assertEqual(kwargs["headers"], {"xi-api-key": "sk_test"})
        self.assertEqual(kwargs["data"], {"model_id": "scribe_v2", "language_code": "en"})
        self.assertIn("file", kwargs["files"])

    def test_language_none_is_omitted(self):
        with mock.patch.object(voice.requests, "post",
                               return_value=FakeResponse(payload={"text": "sit"})) as post:
            voice.transcribe(b"x", "k", None)
        self.assertNotIn("language_code", post.call_args.kwargs["data"])

    def test_http_error_raises_voice_error(self):
        response = FakeResponse(status_code=401, text="invalid key")
        with mock.patch.object(voice.requests, "post", return_value=response):
            with self.assertRaises(VoiceError) as ctx:
                voice.transcribe(b"x", "bad")
        self.assertIn("401", str(ctx.exception))

    def test_network_error_raises_voice_error(self):
        with mock.patch.object(voice.requests, "post",
                               side_effect=requests.ConnectionError("offline")):
            with self.assertRaises(VoiceError):
                voice.transcribe(b"x", "k")


class VoiceCommanderTests(unittest.TestCase):
    def setUp(self):
        self.link = StubLink()
        self.controller = GestureDroneController(
            self.link, GestureMapper(GestureConfig(stable_time=0.0)))
        self.logs = []
        self.commander = VoiceCommander(self.controller, api_key="sk_test",
                                        log=self.logs.append)

    def test_fly_sends_takeoff_through_controller(self):
        self.assertEqual(self.commander.handle_text("Fly."), "fly")
        self.assertEqual(self.link.sent[-1].action, TAKEOFF)

    def test_sit_sends_land_through_controller(self):
        self.assertEqual(self.commander.handle_text("sit"), "sit")
        self.assertEqual(self.link.sent[-1].action, LAND)

    def test_unknown_text_sends_nothing(self):
        self.assertIsNone(self.commander.handle_text("good morning"))
        self.assertEqual(self.link.sent, [])

    def test_start_without_key_is_disabled(self):
        commander = VoiceCommander(self.controller, api_key=None, log=self.logs.append)
        self.assertFalse(commander.start())
        self.assertTrue(any("ELEVENLABS_API_KEY" in line for line in self.logs))

    def test_from_env_reads_key_from_environment(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"ELEVENLABS_API_KEY": "  sk_env  "}):
            commander = VoiceCommander.from_env(self.controller,
                                                env_file=os.path.join(tmp, "missing.env"))
        self.assertEqual(commander.api_key, "sk_env")

    def test_from_env_without_key_anywhere_is_none(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"ELEVENLABS_API_KEY": ""}):
            commander = VoiceCommander.from_env(self.controller,
                                                env_file=os.path.join(tmp, "missing.env"))
        self.assertIsNone(commander.api_key)

    def test_from_env_falls_back_to_dotenv_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# коментар\nELEVENLABS_API_KEY=sk_from_file\n")
            with mock.patch.dict(os.environ, {"ELEVENLABS_API_KEY": ""}):
                commander = VoiceCommander.from_env(self.controller, env_file=path)
        self.assertEqual(commander.api_key, "sk_from_file")


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.link = StubLink()
        self.controller = GestureDroneController(
            self.link, GestureMapper(GestureConfig(stable_time=0.0)))
        self.logs = []
        self.commander = VoiceCommander(self.controller, api_key="sk_test",
                                        log=self.logs.append)

    def test_process_transcribes_and_dispatches(self):
        pcm = b"\x00\x00" * 16000
        with mock.patch.object(voice.requests, "post",
                               return_value=FakeResponse(payload={"text": "fly"})):
            self.commander._process(pcm)
        self.assertEqual(self.link.sent[-1].action, TAKEOFF)

    def test_process_logs_api_error_without_raising(self):
        with mock.patch.object(voice.requests, "post",
                               return_value=FakeResponse(status_code=500, text="boom")):
            self.commander._process(b"\x00\x00" * 16000)
        self.assertEqual(self.link.sent, [])
        self.assertTrue(any("ElevenLabs HTTP 500" in line for line in self.logs))


class ResolveApiKeyTests(unittest.TestCase):
    def test_environment_wins_over_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("ELEVENLABS_API_KEY=sk_file\n")
            with mock.patch.dict(os.environ, {"ELEVENLABS_API_KEY": "sk_env"}):
                self.assertEqual(voice.resolve_api_key(path), "sk_env")

    def test_read_env_file_parses_quotes_and_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# c\n\nA=1\nB=\"two\"\nC='three'\nnoequals\n")
            self.assertEqual(voice.read_env_file(path), {"A": "1", "B": "two", "C": "three"})

    def test_missing_dotenv_returns_empty(self):
        self.assertEqual(voice.read_env_file("/nonexistent/.env"), {})

    def test_banner_on_and_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "none.env")
            with mock.patch.dict(os.environ, {"ELEVENLABS_API_KEY": "sk_x"}):
                self.assertIn("ON", voice.voice_banner_text(False))
            with mock.patch.dict(os.environ, {"ELEVENLABS_API_KEY": ""}), \
                    mock.patch.object(voice, "ENV_FILE_PATH", missing):
                self.assertIn("OFF", voice.voice_banner_text(False))
            self.assertIn("--no-voice", voice.voice_banner_text(True))


class VoicePriorityTests(unittest.TestCase):
    """Голосова команда має переважати над жестами, поки діє (VOICE_HOLD_SECONDS)."""

    def setUp(self):
        self.link = StubLink()
        self.controller = GestureDroneController(
            self.link, GestureMapper(GestureConfig(stable_time=0.0)))

    def _no_hands(self):
        return {"owner_body": {"box": (0, 0, 10, 10)}, "owner_hands": []}

    def test_voice_beats_hover_gesture_within_hold(self):
        self.controller.voice_command(TAKEOFF, label="FLY", now=10.0)
        cmd = self.controller.update(self._no_hands(), FRAME_SHAPE, now=10.2)
        self.assertEqual(cmd.action, TAKEOFF)
        self.assertEqual(cmd.origin, "voice")
        self.assertEqual(self.link.sent[-1].action, TAKEOFF)

    def test_voice_expires_after_hold(self):
        self.controller.voice_command(LAND, label="SIT", now=10.0)
        later = 10.0 + VOICE_HOLD_SECONDS + 0.1
        cmd = self.controller.update(self._no_hands(), FRAME_SHAPE, now=later)
        self.assertEqual(cmd.action, HOVER)

    def test_voice_beats_active_gesture(self):
        self.controller.voice_command(LAND, label="SIT", now=10.0)
        cmd = self.controller.update({"owner_hands": [user_left(make_hand(fingers=(1, 0, 0, 0)))]},
                                     FRAME_SHAPE, now=10.1)
        self.assertEqual(cmd.action, LAND)

    def test_voice_works_even_when_gestures_disabled(self):
        self.controller.enabled = False
        self.controller.voice_command(TAKEOFF, label="FLY", now=10.0)
        cmd = self.controller.update(self._no_hands(), FRAME_SHAPE, now=10.1)
        self.assertEqual(cmd.action, TAKEOFF)
        self.assertEqual(self.link.sent[-1].action, TAKEOFF)


if __name__ == "__main__":
    unittest.main()
