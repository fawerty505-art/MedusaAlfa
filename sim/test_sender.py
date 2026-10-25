#!/usr/bin/env python3
"""
Sends a scripted flight to the drone simulator over WebSocket.

Vector format: [v_x, v_y, v_z, yaw_rate, action]
    v_x  m/s forward(+) / backward(-)      (body frame, North at heading 0)
    v_y  m/s right(+)   / left(-)          (body frame, East at heading 0)
    v_z  m/s up(+)      / down(-)
    yaw_rate  deg/s, positive = clockwise (turn right)
    action 0 HOVER, 1 TAKEOFF (to 2 m), 2 LAND, 3 RESET

Usage:
    python3 sim/server.py            # terminal 1 (or: python3 medusa_webcam.py)
    python3 sim/test_sender.py       # terminal 2 (browser at http://127.0.0.1:8000)
    python3 sim/test_sender.py --loop
"""

import argparse
import asyncio
import json
import math
import sys

try:
    import websockets
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency. Install it with:  python3 -m pip install -r requirements.txt")

HOVER, TAKEOFF, LAND, RESET = 0, 1, 2, 3
RATE_HZ = 20.0
DT = 1.0 / RATE_HZ


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


async def send_phase(ws, name, seconds, fn):
    """Stream fn(t) -> vector at RATE_HZ for `seconds`."""
    print(f"[phase] {name:<26} {seconds:>5.1f}s")
    steps = max(1, int(seconds * RATE_HZ))
    for i in range(steps):
        t = i * DT
        vec = fn(t, seconds)
        await ws.send(json.dumps([round(float(v), 4) for v in vec[:4]] + [int(vec[4])]))
        await asyncio.sleep(DT)


def const(vx=0.0, vy=0.0, vz=0.0, yaw=0.0, action=HOVER):
    return lambda t, T: [vx, vy, vz, yaw, action]


async def flight(ws):
    # Clean start
    await send_phase(ws, "RESET", 0.5, const(action=RESET))
    await send_phase(ws, "TAKEOFF to 2 m", 3.5, const(action=TAKEOFF))
    await send_phase(ws, "hover", 1.0, const())

    # Fly a 6 m square: smooth forward legs, 90 deg turns
    for leg in range(4):
        await send_phase(
            ws, f"square leg {leg + 1}: forward", 3.0,
            lambda t, T: [2.0 * math.sin(math.pi * min(t / T, 1.0)) ** 0.5, 0, 0, 0, HOVER],
        )
        await send_phase(ws, f"square leg {leg + 1}: turn 90 deg", 2.0, const(yaw=45.0))

    # Sideways strafe (roll demonstration)
    await send_phase(ws, "strafe right", 2.0, lambda t, T: [0, 2.0 * smoothstep(t / 0.5), 0, 0, HOVER])
    await send_phase(ws, "strafe left", 2.0, lambda t, T: [0, -2.0 * smoothstep(t / 0.5), 0, 0, HOVER])

    # Climb, circle with yaw coupling, descend
    await send_phase(ws, "climb", 1.5, const(vz=1.0))
    # Circle: speed v, yaw rate w => radius v/w. v=2 m/s, w=40 deg/s -> r ~ 2.9 m
    await send_phase(ws, "circle (full turn)", 9.0, const(vx=2.0, yaw=40.0))
    await send_phase(ws, "descend", 1.5, const(vz=-1.0))
    await send_phase(ws, "hover", 1.0, const())

    # Figure: forward + gentle sinusoidal altitude and yaw weave
    await send_phase(
        ws, "weave", 8.0,
        lambda t, T: [1.5, 0, 0.8 * math.cos(2 * math.pi * t / 4.0), 35.0 * math.sin(2 * math.pi * t / 4.0), HOVER],
    )

    await send_phase(ws, "hover", 1.0, const())
    await send_phase(ws, "LAND", 5.0, const(action=LAND))
    print("[done] flight complete")


async def main(url, loop):
    print(f"Connecting to {url} ...")
    try:
        async with websockets.connect(url) as ws:
            print("Connected. Make sure the browser is open on http://localhost:8000")
            while True:
                await flight(ws)
                if not loop:
                    break
                await asyncio.sleep(1.0)
            # Prove the safety timeout: stop sending; sim should hover within 0.5 s.
            print("Stopped sending (browser should hold hover via 0.5 s safety timeout).")
    except (OSError, websockets.WebSocketException) as exc:
        sys.exit(f"Connection failed: {exc}\nIs 'python3 server.py' running?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone simulator test sender")
    parser.add_argument("--url", default="ws://localhost:8765")
    parser.add_argument("--loop", action="store_true", help="repeat the flight forever")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.url, args.loop))
    except KeyboardInterrupt:
        print("\nInterrupted.")
