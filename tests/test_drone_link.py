"""Наскрізний тест каналу керування: DroneLink -> sim/server.py -> "браузер".

Тест піднімає справжній sim/server.py на вільних портах, підключається до нього
як браузер (звичайний websockets-клієнт) і перевіряє, що вектор жестів доходить
у тому форматі, який очікує app.js, а телеметрія повертається назад.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import unittest

import websockets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medusa.drone_link import DroneLink, coerce_vector  # noqa: E402
from medusa.gestures import DroneCommand, Posture  # noqa: E402
from medusa.paths import SIM_SERVER  # noqa: E402
from medusa.sim_launcher import free_port, port_open  # noqa: E402


class VectorCoercionTests(unittest.TestCase):
    def test_list_passthrough(self):
        self.assertEqual(coerce_vector([1, 0, 0, 0, 0]), [1.0, 0.0, 0.0, 0.0, 0])

    def test_drone_command(self):
        cmd = DroneCommand((1.5, 0.0, 0.0, 12.5, 1), Posture.POINT, "TAKEOFF")
        payload = coerce_vector(cmd)
        self.assertEqual(payload, [1.5, 0.0, 0.0, 12.5, 1])
        self.assertIsInstance(payload[4], int)

    def test_dict_payload(self):
        self.assertEqual(coerce_vector({"cmd": [0, 0, 1, 0, 2]}), [0.0, 0.0, 1.0, 0.0, 2])
        self.assertEqual(coerce_vector({"vector": [0, 0, 1, 0, 2]}), [0.0, 0.0, 1.0, 0.0, 2])

    def test_none_is_hover(self):
        self.assertEqual(coerce_vector(None), [0.0, 0.0, 0.0, 0.0, 0])

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            coerce_vector([1, 0, 0])


class SimRelayIntegrationTests(unittest.TestCase):
    """Реальний sim/server.py у підпроцесі + реальний DroneLink."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(SIM_SERVER):
            raise unittest.SkipTest(f"Немає {SIM_SERVER}")
        cls.http_port = free_port()
        cls.ws_port = free_port()
        cls.process = subprocess.Popen(
            [sys.executable, "-u", str(SIM_SERVER),
             "--host", "127.0.0.1",
             "--http-port", str(cls.http_port),
             "--ws-port", str(cls.ws_port),
             "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        cls.ws_url = f"ws://127.0.0.1:{cls.ws_port}"
        cls.http_url = f"http://127.0.0.1:{cls.http_port}"

        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                raise RuntimeError(f"sim/server.py завершився з кодом {cls.process.returncode}")
            if port_open("127.0.0.1", cls.ws_port) and port_open("127.0.0.1", cls.http_port):
                return
            time.sleep(0.1)
        raise RuntimeError("sim/server.py не піднявся за 15 с")

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        try:
            cls.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:  # pragma: no cover
            cls.process.kill()

    def test_http_serves_the_simulator(self):
        from urllib.request import urlopen

        with urlopen(self.http_url + "/index.html", timeout=5) as response:
            html = response.read().decode("utf-8", "replace")
        self.assertIn("WebGL Drone Simulator", html)

    def test_vector_reaches_the_browser_client(self):
        async def scenario():
            async with websockets.connect(self.ws_url) as browser:
                link = DroneLink(self.ws_url, rate_hz=20.0, verbose=False)
                link.start()
                try:
                    self.assertTrue(link.wait_connected(5.0), "DroneLink не підключився")
                    link.send([1.5, 0.0, 0.0, 0.0, 0])
                    raw = await asyncio.wait_for(browser.recv(), timeout=5.0)
                    self.assertEqual(json.loads(raw), [1.5, 0.0, 0.0, 0.0, 0])
                finally:
                    link.stop()

        asyncio.run(scenario())

    def test_action_vector_keeps_integer_action(self):
        async def scenario():
            async with websockets.connect(self.ws_url) as browser:
                link = DroneLink(self.ws_url, rate_hz=20.0, verbose=False)
                link.start()
                try:
                    self.assertTrue(link.wait_connected(5.0))
                    link.send(DroneCommand((0.0, 0.0, 0.0, 0.0, 1), Posture.POINT, "TAKEOFF"))
                    raw = await asyncio.wait_for(browser.recv(), timeout=5.0)
                    payload = json.loads(raw)
                    self.assertEqual(payload, [0.0, 0.0, 0.0, 0.0, 1])
                    self.assertIsInstance(payload[4], int)
                finally:
                    link.stop()

        asyncio.run(scenario())

    def test_telemetry_is_read_by_the_link(self):
        async def scenario():
            async with websockets.connect(self.ws_url) as browser:
                link = DroneLink(self.ws_url, rate_hz=20.0, verbose=False)
                link.start()
                try:
                    self.assertTrue(link.wait_connected(5.0))
                    await browser.send(json.dumps({"telemetry": {
                        "x": 1.0, "y": 2.0, "z": -3.0, "heading": 90.0, "status": "FLYING"}}))
                    deadline = time.monotonic() + 5.0
                    while link.telemetry is None and time.monotonic() < deadline:
                        await asyncio.sleep(0.05)
                    telemetry = link.telemetry
                    self.assertIsNotNone(telemetry)
                    self.assertEqual(telemetry["status"], "FLYING")
                    self.assertEqual(telemetry["heading"], 90.0)
                finally:
                    link.stop()

        asyncio.run(scenario())

    def test_stale_vector_is_not_repeated(self):
        """Регресія: раніше останній вектор слався вічно, тож зависання/вимкнення 'G' не зупиняло дрон."""
        async def scenario():
            async with websockets.connect(self.ws_url) as browser:
                link = DroneLink(self.ws_url, rate_hz=40.0, verbose=False, max_age=0.2)
                link.start()
                try:
                    self.assertTrue(link.wait_connected(5.0))
                    link.send([1.5, 0.0, 0.0, 0.0, 0])
                    await asyncio.wait_for(browser.recv(), timeout=5.0)   # свіжий вектор дійшов
                    await asyncio.sleep(0.5)                              # він застарів
                    while True:                                           # вичерпуємо те, що встигло дійти
                        try:
                            await asyncio.wait_for(browser.recv(), timeout=0.05)
                        except asyncio.TimeoutError:
                            break
                    with self.assertRaises(asyncio.TimeoutError):         # далі — тиша
                        await asyncio.wait_for(browser.recv(), timeout=0.6)
                finally:
                    link.stop()

        asyncio.run(scenario())

    def test_link_reports_offline_when_simulator_is_gone(self):
        dead_port = free_port()
        link = DroneLink(f"ws://127.0.0.1:{dead_port}", reconnect_delay=0.2,
                         open_timeout=0.5, verbose=False)
        link.start()
        try:
            self.assertFalse(link.wait_connected(1.0))
            self.assertEqual(link.status_text(), "OFFLINE")
        finally:
            link.stop()


if __name__ == "__main__":
    unittest.main()

