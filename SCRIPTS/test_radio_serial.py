#!/usr/bin/env python3
"""Send test RC channel values to an EdgeTX radio over USB serial.

Install the optional dependency with: python -m pip install pyserial
Run with: python scripts/test_radio_serial.py --port COM10
"""

import argparse
import sys
import time


IDLE_CHANNELS = (1500, 1500, 1000, 1500)
CLIMB_CHANNELS = (1500, 1500, 1400, 1500)
BAUD_RATE = 115200
FRAMES_PER_STEP = 50
FRAME_INTERVAL_SECONDS = 0.02


def send_channels(connection, channels):
    line = ",".join(str(value) for value in channels) + "\n"
    connection.write(line.encode("ascii"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        default="COM10",
        help="radio serial port (default: COM10; check Device Manager)",
    )
    args = parser.parse_args()

    try:
        import serial
    except ImportError:
        print(
            "Missing dependency. Install it with: python -m pip install pyserial",
            file=sys.stderr,
        )
        return 1

    try:
        connection = serial.Serial(
            args.port,
            BAUD_RATE,
            timeout=0.01,
            dsrdtr=False,
            rtscts=False,
        )
    except serial.SerialException as error:
        print(f"Could not open {args.port}: {error}", file=sys.stderr)
        print(
            "Check Device Manager for the correct port and resolve any "
            "USB serial driver warning.",
            file=sys.stderr,
        )
        return 1

    print(f"Streaming test channel data to {args.port} at {BAUD_RATE} baud.")
    print("The radio should show T:-100 then T:-20; R, P, and Y should stay at 0.")
    print("Keep RF disabled and do not connect an aircraft for this screen test.")
    try:
        while True:
            print("Sending idle throttle (1000 us; expected T:-100)")
            for _ in range(FRAMES_PER_STEP):
                send_channels(connection, IDLE_CHANNELS)
                time.sleep(FRAME_INTERVAL_SECONDS)

            print("Sending climb/hover throttle (1400 us; expected T:-20)")
            for _ in range(FRAMES_PER_STEP):
                send_channels(connection, CLIMB_CHANNELS)
                time.sleep(FRAME_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopping; sending idle throttle.")
    finally:
        try:
            if connection.is_open:
                send_channels(connection, IDLE_CHANNELS)
        finally:
            connection.close()
    print("Stopped safely.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
