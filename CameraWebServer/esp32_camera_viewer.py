import argparse
from collections import deque
import socket
import struct
import sys
import time


MAX_FRAME_SIZE = 5 * 1024 * 1024


def recv_exact(connection, length):
    data = bytearray()
    while len(data) < length:
        chunk = connection.recv(length - len(data))
        if not chunk:
            raise ConnectionError("ESP32 closed the connection.")
        data.extend(chunk)
    return bytes(data)


def main():
    parser = argparse.ArgumentParser(
        description="Connect to the ESP32 camera stream and show live frames."
    )
    parser.add_argument(
        "ip",
        nargs="?",
        default="10.21.246.241",
        help="ESP32 IP address (default: last address printed by the board)",
    )
    parser.add_argument("--port", type=int, default=80, help="ESP32 port (default: 80)")
    args = parser.parse_args()

    try:
        import cv2
        import numpy
    except ImportError:
        print(
            "Missing viewer packages. Install them with:\n"
            "  py -m pip install opencv-python numpy",
            file=sys.stderr,
        )
        return 1

    try:
        print(f"Connecting to {args.ip}:{args.port} ...")
        with socket.create_connection((args.ip, args.port), timeout=5) as connection:
            connection.settimeout(15)
            connection.sendall(b"START\n")
            print("Connected; waiting for camera frames. Press Q in the video window to quit.")

            frame_times = deque()
            last_report = time.monotonic()
            frame_count = 0
            received_bytes = 0
            receive_seconds = 0.0
            decode_seconds = 0.0
            display_seconds = 0.0
            while True:
                frame_started = time.monotonic()
                frame_size = struct.unpack(">I", recv_exact(connection, 4))[0]
                if frame_size == 0 or frame_size > MAX_FRAME_SIZE:
                    raise ValueError(
                        f"Invalid frame size {frame_size} bytes; "
                        "check that this is the ESP32 camera server."
                    )

                jpeg_data = recv_exact(connection, frame_size)
                received_at = time.monotonic()
                image = cv2.imdecode(
                    numpy.frombuffer(jpeg_data, dtype=numpy.uint8),
                    cv2.IMREAD_COLOR,
                )
                decoded_at = time.monotonic()
                if image is None:
                    print(f"Received {frame_size} bytes, but JPEG decoding failed.")
                    continue

                cv2.imshow("ESP32 camera - press Q to quit", image)
                key = cv2.waitKey(1) & 0xFF
                displayed_at = time.monotonic()

                frame_count += 1
                received_bytes += frame_size
                receive_seconds += received_at - frame_started
                decode_seconds += decoded_at - received_at
                display_seconds += displayed_at - decoded_at

                now = displayed_at
                frame_times.append(now)
                while frame_times and now - frame_times[0] > 3:
                    frame_times.popleft()

                if now - last_report >= 2:
                    if len(frame_times) > 1:
                        rolling_fps = (len(frame_times) - 1) / (
                            frame_times[-1] - frame_times[0]
                        )
                        interval = now - last_report
                        interval_fps = frame_count / interval
                        average_kb = received_bytes / frame_count / 1024
                        average_receive_ms = receive_seconds / frame_count * 1000
                        average_decode_ms = decode_seconds / frame_count * 1000
                        average_display_ms = display_seconds / frame_count * 1000
                        print(
                            f"{image.shape[1]}x{image.shape[0]} | "
                            f"{rolling_fps:.1f} FPS rolling, {interval_fps:.1f} FPS interval | "
                            f"{average_kb:.1f} KB/frame | "
                            f"receive {average_receive_ms:.1f} ms, "
                            f"decode {average_decode_ms:.1f} ms, "
                            f"display {average_display_ms:.1f} ms"
                        )
                    else:
                        print("Receiving video: waiting for more frames")
                    last_report = now
                    frame_count = 0
                    received_bytes = 0
                    receive_seconds = 0.0
                    decode_seconds = 0.0
                    display_seconds = 0.0

                if key in (ord("q"), ord("Q")):
                    break
    except (OSError, ConnectionError, ValueError, struct.error) as error:
        print(f"Stream error: {error}", file=sys.stderr)
        print(
            "Check the current IP printed by the ESP32 and make sure the laptop "
            "and ESP32 can communicate on the same network.",
            file=sys.stderr,
        )
        return 1
    finally:
        if "cv2" in locals():
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
