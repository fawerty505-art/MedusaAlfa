#!/usr/bin/env python3
"""
=============================================================================
  MedusaAlfa — Голосові команди (ElevenLabs Speech-to-Text)
  Push-to-talk: зажати SPACE -> сказати "fly" (зліт) або "sit" (посадка) -> відпустити
=============================================================================

Потік:
    SPACE натиснуто   -> запис мікрофона (16 кГц, моно, PCM16)
    SPACE відпущено   -> WAV -> POST https://api.elevenlabs.io/v1/speech-to-text
                      -> текст: "fly" = TAKEOFF, "sit" = LAND
                      -> GestureDroneController.voice_command(...)

Ключ API читається ТІЛЬКИ зі змінної оточення ELEVENLABS_API_KEY
(у код і git він не потрапляє).

Залежності, що імпортуються ліниво (тести і --no-voice працюють без них):
    sounddevice — запис з мікрофона (потрібен PortAudio)
    pynput      — глобальний hotkey SPACE (macOS: дозвіл "Input Monitoring")
=============================================================================
"""

from __future__ import annotations

import io
import os
import re
import threading
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

from .gestures import ACTION_NAMES, LAND, TAKEOFF
from .paths import ENV_FILE_PATH

ELEVENLABS_API_KEY_ENV = "ELEVENLABS_API_KEY"
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVENLABS_MODEL_ID = "scribe_v2"

SAMPLE_RATE = 16000
MIN_SECONDS = 0.3      # коротше — випадкове натискання SPACE, ігноруємо
MAX_SECONDS = 5.0      # довше — обрізаємо (голосові команди короткі)
REQUEST_TIMEOUT = 15.0

#: голосова команда -> дія симулятора
VOICE_COMMANDS: Dict[str, int] = {"fly": TAKEOFF, "sit": LAND}

_WORD_RE = re.compile(r"[a-z]+")


class VoiceError(RuntimeError):
    """Помилка запиту до ElevenLabs."""


def parse_voice_command(text: str) -> Optional[str]:
    """
    'Fly.' -> 'fly', 'SIT!' -> 'sit', 'hello' -> None.
    Якщо команд немає або їх дві ('fly sit') — None (неоднозначно).
    """
    found = {word for word in _WORD_RE.findall((text or "").lower()) if word in VOICE_COMMANDS}
    if len(found) == 1:
        return found.pop()
    return None


def make_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE, channels: int = 1) -> bytes:
    """Упаковує сирі PCM16 у WAV-контейнер (в пам'яті, без файлів на диску)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def transcribe(wav_bytes: bytes, api_key: str, language_code: Optional[str] = "en",
               timeout: float = REQUEST_TIMEOUT) -> str:
    """Надсилає WAV в ElevenLabs Speech-to-Text і повертає розпізнаний текст."""
    data = {"model_id": ELEVENLABS_MODEL_ID}
    if language_code:
        data["language_code"] = language_code
    try:
        response = requests.post(
            ELEVENLABS_STT_URL,
            headers={"xi-api-key": api_key},
            data=data,
            files={"file": ("command.wav", wav_bytes, "audio/wav")},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise VoiceError(f"немає зв'язку з ElevenLabs: {exc}") from exc

    if response.status_code != 200:
        raise VoiceError(f"ElevenLabs HTTP {response.status_code}: {response.text[:200]}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise VoiceError("ElevenLabs повернув не JSON") from exc
    return str(payload.get("text", "")).strip()


def read_env_file(path) -> Dict[str, str]:
    """Читає простий .env (KEY=VALUE, рядки з '#' — коментарі). Немає файлу -> порожньо."""
    values: Dict[str, str] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_api_key(env_file=None) -> Optional[str]:
    """Ключ ElevenLabs: спочатку змінна оточення, потім файл .env у корені проєкту."""
    key = os.environ.get(ELEVENLABS_API_KEY_ENV, "").strip()
    if not key:
        path = env_file if env_file is not None else ENV_FILE_PATH
        key = read_env_file(path).get(ELEVENLABS_API_KEY_ENV, "").strip()
    return key or None


def voice_banner_text(disabled: bool) -> str:
    """Рядок стану голосу для банера CLI."""
    if disabled:
        return "OFF (--no-voice)"
    if resolve_api_key():
        return f"ON  (SPACE: fly / sit, ElevenLabs {ELEVENLABS_MODEL_ID})"
    return f"OFF (не задано {ELEVENLABS_API_KEY_ENV} ні в оточенні, ні в .env)"


class VoiceCommander:
    """Push-to-talk: SPACE зажато -> запис, відпущено -> розпізнавання -> команда дрона."""

    def __init__(
        self,
        controller: Any,
        api_key: Optional[str] = None,
        language_code: Optional[str] = "en",
        log: Callable[[str], None] = print,
    ) -> None:
        self.controller = controller
        self.api_key = api_key or None
        self.language_code = language_code
        self.log = log

        self._listener: Any = None
        self._space_key: Any = None
        self._stream: Any = None
        self._frames: List[bytes] = []
        self._samples = 0
        self._recording = False

    @classmethod
    def from_env(cls, controller: Any, env_file=None, **kwargs) -> "VoiceCommander":
        return cls(controller, api_key=resolve_api_key(env_file), **kwargs)

    # ------------------------------------------------------------------ життя
    def start(self) -> bool:
        """Запускає глобальний слухач SPACE. False — якщо голос не може працювати."""
        if not self.api_key:
            self.log(f"⚠ Голос вимкнено: не задано {ELEVENLABS_API_KEY_ENV} (оточення або .env)")
            return False
        try:
            from pynput import keyboard  # ліниво: pynput потребує дозволу macOS
        except Exception as exc:  # ImportError або помилка доступу до клавіатури macOS
            self.log(f"⚠ Голос вимкнено: pynput недоступний ({exc})")
            return False

        self._space_key = keyboard.Key.space
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()
        self.log("🎙  Голос: зажми SPACE, скажи 'fly' (зліт) або 'sit' (посадка), відпусти")
        return True

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # pragma: no cover - defensive
                pass
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.stop()

    # --------------------------------------------------------------- hotkey
    def _on_press(self, key) -> None:
        if key != self._space_key or self._recording:   # повторне натискання (auto-repeat) ігноруємо
            return
        self._start_recording()

    def _on_release(self, key) -> None:
        if key != self._space_key or not self._recording:
            return
        self._recording = False
        pcm = self._stop_recording()
        if pcm is not None:
            threading.Thread(target=self._process, args=(pcm,), name="voice-stt", daemon=True).start()

    # --------------------------------------------------------------- запис
    def _start_recording(self) -> None:
        try:
            import sounddevice as sd  # ліниво: потребує PortAudio
            self._frames = []
            self._samples = 0
            self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                          callback=self._on_audio)
            self._stream.start()
        except Exception as exc:  # різні типи помилок PortAudio / відсутній мікрофон
            self._stream = None
            self.log(f"⚠ Мікрофон недоступний: {exc}")
            return
        self._recording = True
        self.log("🔴 Запис... (відпусти SPACE, щоб надіслати)")

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if self._samples >= MAX_SECONDS * SAMPLE_RATE:
            return
        self._frames.append(indata.tobytes())
        self._samples += frames

    def _stop_recording(self) -> Optional[bytes]:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()      # чекає, поки callback завершиться
                stream.close()
            except Exception:  # pragma: no cover - defensive
                pass
        pcm = b"".join(self._frames)
        self._frames = []
        seconds = len(pcm) / 2 / SAMPLE_RATE
        if seconds < MIN_SECONDS:
            self.log(f"… запис {seconds:.2f} c — занадто коротко, ігнорую")
            return None
        self.log(f"⏹  Запис {seconds:.1f} c -> розпізнаю...")
        return pcm

    # ----------------------------------------------------------- обробка
    def _process(self, pcm: bytes) -> None:
        try:
            text = transcribe(make_wav(pcm), self.api_key, self.language_code)
        except VoiceError as exc:
            self.log(f"❌ Голос: {exc}")
            return
        self.handle_text(text)

    def handle_text(self, text: str) -> Optional[str]:
        """Перетворює розпізнаний текст на команду дрона. Повертає слово-команду або None."""
        word = parse_voice_command(text)
        if word is None:
            self.log(f"🗣  почуто: {text!r} — команди немає (скажи fly або sit)")
            return None
        action = VOICE_COMMANDS[word]
        self.controller.voice_command(action, label=word.upper())
        self.log(f"🎙  '{word}' -> {ACTION_NAMES[action]}")
        return word
