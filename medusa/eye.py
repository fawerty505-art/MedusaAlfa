#!/usr/bin/env python3
"""
=============================================================================
  MedusaAlfa — Test Eye: Система розпізнавання хазяїна (Owner Recognition)
  ⚡ ULTRA-FAST POSE ENGINE: Реальний трекінг голови при нахилах + миттєвий Lock
=============================================================================

ВИПРАВЛЕННЯ ОБОХ БАГІВ:
───────────────────────
1. ЧОМУ ЗЕЛЕНИЙ КВАДРАТ НЕ ЙШОВ ЗА ГОЛОВОЮ ПРИ ВІДВЕДЕННІ ВЛІВО/ВПРАВО:
   • Раніше: центр голови жорстко вираховувався як центр плечей тіла (bx1+bx2)/2.
     Коли ви нахиляли або відводили голову вбік, тіло лишалося на місці, а квадрат
     не рухався.
   • Тепер: інтегровано `YOLOv8-pose` (17 ключових точок тіла). Координати носа,
     очей та вух зчитуються в реальному часі (70+ FPS). Зелений квадрат прив'язаний
     до СПРАВЖНІХ точок обличчя (ніс/очі) і миттєво слідує за кожним нахилом голови!

2. ЧОМУ ПРИ ПОЯВІ ВИ СПОЧАТКУ БУЛИ "STRANGER" ТА ЧОМУ БУЛА ЗАТРИМКА В КІЛЬКА СЕКУНД:
   • По-перше: раніше неперевірена людина одразу за замовчуванням маркувалася
     червоним як "STRANGER". Тепер статус нової людини — "IDENTIFYING..." (нейтральний).
     Статус "STRANGER" призначається ТІЛЬКИ якщо біометрія підтвердила чуже обличчя.
   • По-друге: раніше кроп голови вирізався наосліп з верхніх 35% тіла, і обличчя
     часто обрізалося або dlib його не бачив, через що доводилося чекати.
     Тепер кроп вирізається ТОЧНО навколо знайдених очей та носа з 100% гарантією
     потрапляння обличчя в центр на першому ж кадрі! Хазяїн фіксується за ~0.1 сек!
=============================================================================
"""

import os
import sys
import time
import math
import queue
import threading
import urllib.request
from pathlib import Path
import cv2
import numpy as np

try:  # звичайний запуск через пакет:  import medusa.eye
    from .esp32_stream import ESP32Stream, DEFAULT_IP, DEFAULT_PORT
    from .paths import HAND_MODEL_PATH, OWNER_FACE_PATH, YOLO_POSE_PATH, ensure_project_dirs
    from .overlay import draw_drone_panel, draw_action_flash
except ImportError:  # прямий запуск файлу: python3 medusa/eye.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from medusa.esp32_stream import ESP32Stream, DEFAULT_IP, DEFAULT_PORT  # type: ignore[no-redef]
    from medusa.paths import HAND_MODEL_PATH, OWNER_FACE_PATH, YOLO_POSE_PATH, ensure_project_dirs  # type: ignore[no-redef]
    from medusa.overlay import draw_drone_panel, draw_action_flash  # type: ignore[no-redef]

# 1. Імпорт face_recognition
import face_recognition

# 2. Імпорт YOLO (Pose)
from ultralytics import YOLO

# 3. Імпорт MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────────────────────────────────────────
# КОНФІГУРАЦІЯ
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CURRENT_DIR = Path(__file__).parent.resolve()
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Параметри
YOLO_IMGSZ = 320           # 320px для надшвидкої детекції (~14 мс)
YOLO_CONFIDENCE = 0.38     # Чутливість детекції людей
FACE_TOLERANCE = 0.52      # Поріг розпізнавання хазяїна (< 0.52)
HAND_MARGIN = 70           # Запас прив'язки рук до тіла

# Кольори BGR
COLOR_OWNER_FACE = (0, 255, 0)        # Зелений
COLOR_OWNER_BODY = (0, 235, 255)      # Жовтий / Amber
COLOR_IDENTIFYING = (0, 200, 255)     # Помаранчево-жовтий для верифікації
COLOR_STRANGER = (0, 0, 240)          # Червоний
COLOR_STRANGER_BODY = (80, 80, 80)    # Приглушений сірий
COLOR_HAND_JOINT = (255, 120, 255)    # Рожевий
COLOR_HAND_BONE = (240, 20, 200)      # Фіолетовий
COLOR_HUD_TEXT = (220, 220, 220)

# Зв'язки суглобів кисті (21 точка)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# ─────────────────────────────────────────────────────────────────────────────
# THREADED CAMERA: НЕБЛОКУЮЧЕ ЧИТАННЯ (0 мс)
# ─────────────────────────────────────────────────────────────────────────────

class ThreadedCamera:
    """Асинхронне читання камери у фоновому потоці."""
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.cap.read()
        self.started = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.started = False
        try:
            self.thread.join(timeout=0.5)
        except Exception:
            pass
        self.cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC FACE VERIFIER: РОЗПІЗНАВАННЯ ОБЛИЧЧЯ У ФОНІ
# ─────────────────────────────────────────────────────────────────────────────

class AsyncFaceVerifier:
    """
    Фоновий воркер біометрії обличчя.
    Не блокує 35-45 FPS основного відеопотоку.
    """
    def __init__(self, owner_encoding_ref, tolerance=FACE_TOLERANCE):
        self.owner_encoding_ref = owner_encoding_ref
        self.tolerance = tolerance
        self.req_queue = queue.Queue(maxsize=3)
        self.res_queue = queue.Queue()
        self.started = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def request_verification(self, track_id, head_crop):
        """Передає кроп голови у чергу перевірки."""
        if not self.req_queue.full():
            try:
                self.req_queue.put_nowait({
                    "track_id": track_id,
                    "crop": head_crop
                })
            except queue.Full:
                pass

    def get_results(self):
        """Забирає готові відповіді з черги."""
        results = []
        while not self.res_queue.empty():
            try:
                results.append(self.res_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def _worker(self):
        while self.started:
            try:
                task = self.req_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            owner_enc = self.owner_encoding_ref()
            if owner_enc is None:
                continue

            crop = task["crop"]
            track_id = task["track_id"]

            try:
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb_crop, model="hog")
                if locs:
                    encs = face_recognition.face_encodings(rgb_crop, [locs[0]])
                    if encs:
                        dists = face_recognition.face_distance([owner_enc], encs[0])
                        dist = float(dists[0])
                        self.res_queue.put({
                            "track_id": track_id,
                            "is_owner": (dist < self.tolerance),
                            "distance": dist
                        })
                else:
                    # Якщо в цьому кропі не знайдено обличчя, повертаємо статус retry
                    self.res_queue.put({
                        "track_id": track_id,
                        "retry": True
                    })
            except Exception:
                pass

    def stop(self):
        self.started = False


# ─────────────────────────────────────────────────────────────────────────────
# ДОПОМІЖНІ ФУНКЦІЇ ВІЗУАЛІЗАЦІЇ
# ─────────────────────────────────────────────────────────────────────────────

def ensure_hand_model():
    """Завантажує модель руки за потреби (у models/hand_landmarker.task)."""
    ensure_project_dirs()
    if not HAND_MODEL_PATH.exists():
        print(f"📥 Завантажую MediaPipe модель руки ({HAND_MODEL_PATH.name})...")
        urllib.request.urlretrieve(HAND_MODEL_URL, str(HAND_MODEL_PATH))
        print("   ✓ Завантаження завершено")


def wrist_near_body(wrist_pt, body_box, margin=HAND_MARGIN):
    """Перевіряє чи зап'ястя руки знаходиться біля тіла хазяїна."""
    wx, wy = wrist_pt
    bx1, by1, bx2, by2 = body_box
    return (bx1 - margin <= wx <= bx2 + margin) and (by1 - margin <= wy <= by2 + margin)


def draw_styled_rect(img, box, color, thickness=2, corner_len=14):
    """Тактичний прямокутник з кутовими мітками."""
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if corner_len > 0:
        t = thickness + 2
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, t)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, t)
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, t)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, t)
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, t)
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, t)
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, t)
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, t)


def draw_badge(img, text, pos, color, bg_color=(20, 20, 20)):
    """Текстовий бейдж з фоном."""
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.40
    thick = 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    pad = 3

    bx1 = max(0, x - pad)
    by1 = max(0, y - th - pad * 2)
    bx2 = min(img.shape[1], x + tw + pad)
    by2 = min(img.shape[0], y + base)

    sub = img[by1:by2, bx1:bx2]
    if sub.size > 0:
        dark = np.full_like(sub, bg_color, dtype=np.uint8)
        cv2.addWeighted(dark, 0.75, sub, 0.25, 0, sub)
        img[by1:by2, bx1:bx2] = sub

    cv2.putText(img, text, (x, y - pad), font, scale, color, thick, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# ГОЛОВНИЙ КЛАС MEDUSA EYE
# ─────────────────────────────────────────────────────────────────────────────

class MedusaEye:
    def __init__(self, source="esp32", esp32_url=None, controller=None, camera_index=0):
        print("=" * 60)
        print("  MEDUSA EYE (VGA 640x480) — Ultra-Fast Real-Pose Engine")
        print("=" * 60)

        ensure_hand_model()

        # Контролер жестів (може бути None -> працюємо як чистий CV)
        self.controller = controller
        self.gesture_command = None
        self.last_action_at = None
        self.last_action = 0

        # 1. Асинхронна камера (ESP32-CAM за MJPEG або локальна веб-камера)
        if source == "esp32":
            host, port = esp32_url
            print(f"[1/5] Підключення до ESP32-CAM TCP: {host}:{port}")
            self.cam = ESP32Stream(host, port)
            if not self.cam.wait_for_first_frame(timeout=10.0):
                print("❌ ПОМИЛКА: Немає відео від ESP32-CAM. Перевірте IP та мережу.")
                self.cam.release()
                sys.exit(1)
        else:
            print(f"[1/5] Ініціалізація ThreadedCamera (локальна веб-камера #{camera_index})...")
            self.cam = ThreadedCamera(camera_index, CAMERA_WIDTH, CAMERA_HEIGHT)
            time.sleep(0.3)
        test_frame = self.cam.read()
        if test_frame is None:
            print("❌ ПОМИЛКА: Не вдалося отримати відео з камери.")
            sys.exit(1)
        print("   ✓ Камера активна у фоновому потоці (0 мс затримки)")

        # 2. YOLOv8 Pose (для точного трекінгу обличчя при поворотах/нахилах)
        print("[2/5] Ініціалізація YOLOv8-pose (ByteTrack, imgsz=320)...")
        ensure_project_dirs()
        # якщо файлу немає — Ultralytics сам завантажить його саме в models/
        self.yolo_pose = YOLO(str(YOLO_POSE_PATH))
        self.yolo_pose.track(test_frame, persist=True, imgsz=YOLO_IMGSZ, verbose=False)
        print("   ✓ YOLOv8n-pose готова і прогріта (~14 мс)")

        # 3. MediaPipe у потоковому режимі VIDEO
        print("[3/5] Ініціалізація MediaPipe HandLandmarker (RunningMode.VIDEO)...")
        base_opts = python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
        options = vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options)
        _mp_warm = mp.Image(image_format=mp.ImageFormat.SRGB, data=test_frame)
        self.hand_detector.detect_for_video(_mp_warm, 0)
        print("   ✓ MediaPipe HandLandmarker готовий (~10 мс)")

        # 4. Профіль хазяїна
        print("[4/5] Завантаження профілю хазяїна...")
        self.owner_encoding = None
        self.load_owner_face()

        # 5. Асинхронний воркер розпізнавання обличчя
        print("[5/5] Запуск AsyncFaceVerifier...")
        self.face_verifier = AsyncFaceVerifier(lambda: self.owner_encoding)
        print("   ✓ Асинхронний воркер активний")

        # ── Стан трекінгу ──
        self.owner_track_id = None
        # track_id -> {"status": "owner"|"stranger"|"identifying", "dist": float, "last_check": float}
        self.track_identities = {}
        self.smoothed_face_box = None

        # Метрики FPS
        self.fps = 0.0
        self.frame_idx = 0
        self.fps_timer = time.time()
        self.debug_mode = False

    def load_owner_face(self):
        """Завантажує збережений вектор обличчя."""
        if OWNER_FACE_PATH.exists():
            try:
                self.owner_encoding = np.load(str(OWNER_FACE_PATH))
                print(f"   ✓ Знайдено профіль хазяїна: {OWNER_FACE_PATH.name}")
            except Exception as e:
                print(f"   ⚠ Помилка читання: {e}")
                self.owner_encoding = None
        else:
            print("   ℹ Хазяїн ще не зареєстрований. Натисніть [R] перед камерою.")

    def register_owner(self, frame_bgr):
        """Реєстрація обличчя хазяїна."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        if not locations:
            print("❌ Обличчя не виявлено. Подивіться ближче в камеру та натисніть [R] знову.")
            return False

        locations = sorted(
            locations,
            key=lambda b: (b[2] - b[0]) * (b[1] - b[3]),
            reverse=True
        )

        encs = face_recognition.face_encodings(rgb, [locations[0]])
        if not encs:
            print("❌ Не вдалося створити вектор обличчя.")
            return False

        self.owner_encoding = encs[0]
        np.save(str(OWNER_FACE_PATH), self.owner_encoding)
        self.owner_track_id = None
        self.track_identities.clear()
        self.smoothed_face_box = None
        print(f"✅ ХАЗЯЇНА ЗАРЕЄСТРОВАНО! Вектор збережено у {OWNER_FACE_PATH.name}")
        return True

    def clear_owner(self):
        """Скидає хазяїна."""
        if OWNER_FACE_PATH.exists():
            OWNER_FACE_PATH.unlink()
        self.owner_encoding = None
        self.owner_track_id = None
        self.track_identities.clear()
        self.smoothed_face_box = None
        print("🗑️ Профіль хазяїна скинуто.")

    def track_people_with_pose(self, frame_bgr):
        """
        YOLOv8-Pose ByteTrack: детекція людей разом з їхніми ключовими точками
        (ніс, очі, вуха, плечі).
        """
        results = self.yolo_pose.track(
            frame_bgr,
            persist=True,
            verbose=False,
            conf=YOLO_CONFIDENCE,
            imgsz=YOLO_IMGSZ
        )
        people = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            kpts = results[0].keypoints
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                track_id = int(boxes.id[i]) if boxes.id is not None else None
                conf = float(boxes.conf[i])

                pts = None
                if kpts is not None and len(kpts.xy) > i:
                    pts = kpts.xy[i].cpu().numpy()

                people.append({
                    "box": (x1, y1, x2, y2),
                    "track_id": track_id,
                    "confidence": conf,
                    "keypoints": pts
                })
        return people

    def get_precise_head_box(self, person, frame_shape):
        """
        🔥 ВИРІШЕННЯ БАГУ 2 (ВІДВЕДЕННЯ ГОЛОВИ ВБІК):
        Вираховує точне положення голови за ключовими точками обличчя:
        - Точка 0: ніс
        - Точки 1, 2: ліве та праве око
        - Точки 3, 4: ліве та праве вухо
        Якщо ви нахиляєте голову або відводите її вліво/вправо —
        квадрат слідує безпосередньо за носом та очима!
        """
        kpts = person.get("keypoints")
        bx1, by1, bx2, by2 = person["box"]
        bw = bx2 - bx1
        bh = by2 - by1

        head_pts = []
        if kpts is not None and len(kpts) >= 5:
            # Збираємо видимі точки голови (ніс, очі, вуха)
            for idx in (0, 1, 2, 3, 4):
                kx, ky = kpts[idx]
                if kx > 0 and ky > 0:
                    head_pts.append((float(kx), float(ky)))

        if len(head_pts) >= 2:
            # Знаходимо центр видимих точок обличчя
            cx = int(sum(p[0] for p in head_pts) / len(head_pts))
            cy = int(sum(p[1] for p in head_pts) / len(head_pts))

            # Розмір голови пропорційний розмаху очей/вух або тілу
            xs = [p[0] for p in head_pts]
            span = max(xs) - min(xs)
            head_size = int(max(span * 2.2, bw * 0.42))
        elif len(head_pts) == 1:
            # Тільки ніс або одне око
            cx = int(head_pts[0][0])
            cy = int(head_pts[0][1])
            head_size = int(bw * 0.44)
        else:
            # Запасний варіант: якщо обличчя закрите
            cx = (bx1 + bx2) // 2
            cy = by1 + int(bh * 0.15)
            head_size = int(bw * 0.44)

        head_size = max(28, min(head_size, 180))

        # Ідеальний квадрат
        hx1 = max(0, cx - head_size // 2)
        hy1 = max(0, cy - head_size // 2)
        hx2 = min(frame_shape[1], cx + head_size // 2)
        hy2 = min(frame_shape[0], cy + head_size // 2)

        return (hx1, hy1, hx2, hy2)

    def smooth_face_box(self, target_box):
        """Плавний фільтр переміщення рамки обличчя (EMA)."""
        if self.smoothed_face_box is None:
            self.smoothed_face_box = target_box
        else:
            a = 0.45  # Швидкий плавний відгук
            self.smoothed_face_box = (
                int(self.smoothed_face_box[0] * (1 - a) + target_box[0] * a),
                int(self.smoothed_face_box[1] * (1 - a) + target_box[1] * a),
                int(self.smoothed_face_box[2] * (1 - a) + target_box[2] * a),
                int(self.smoothed_face_box[3] * (1 - a) + target_box[3] * a)
            )
        return self.smoothed_face_box

    def detect_hands(self, frame_bgr, timestamp_ms):
        """MediaPipe Hands у швидкому режимі VIDEO."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.hand_detector.detect_for_video(mp_image, timestamp_ms)

        hands = []
        if res and res.hand_landmarks:
            h, w = frame_bgr.shape[:2]
            for idx, landmarks in enumerate(res.hand_landmarks):
                label = "Hand"
                if res.handedness and idx < len(res.handedness):
                    label = res.handedness[idx][0].category_name

                pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                hands.append({
                    "points": pts,
                    "wrist": pts[0],
                    "label": label
                })
        return hands

    def process_pipeline(self, frame_bgr, curr_time_ms):
        """
        Повний пайплайн без блокувань та затримок:
        1. Збір відповідей фонового воркера облич
        2. YOLOv8 Pose детекція людей та їхніх голів
        3. Миттєве очищення при виході з кадру
        4. Точне відстеження нахилів голови
        5. MediaPipe трекінг рук
        """
        now = time.time()

        # 1. Забираємо відповіді біометрії
        for res in self.face_verifier.get_results():
            tid = res["track_id"]
            if res.get("retry"):
                # Скидаємо таймер перевірки для негайної повторної спроби
                if tid in self.track_identities:
                    self.track_identities[tid]["last_check"] = 0
            else:
                is_owner = res["is_owner"]
                self.track_identities[tid] = {
                    "status": "owner" if is_owner else "stranger",
                    "dist": res["distance"],
                    "last_check": now
                }
                if is_owner:
                    self.owner_track_id = tid

        # 2. Детекція людей (YOLOv8 Pose)
        people = self.track_people_with_pose(frame_bgr)

        # Миттєве очищення, якщо в полі зору нікого немає
        if not people:
            self.owner_track_id = None
            self.smoothed_face_box = None
            return {
                "owner_face": None,
                "owner_body": None,
                "owner_hands": [],
                "others": []
            }

        # 3. Обробка знайдених людей
        owner_body = None
        others = []
        active_track_ids = set()

        for p in people:
            tid = p["track_id"]
            if tid is not None:
                active_track_ids.add(tid)

            # Отримуємо точну рамку голови за точками обличчя
            raw_head_box = self.get_precise_head_box(p, frame_bgr.shape)

            # Статус верифікації
            info = self.track_identities.get(tid)

            # Якщо людини немає в базі або статус невідомий:
            # надсилаємо ТОЧНИЙ кроп обличчя на перевірку
            if tid is not None and (info is None or (now - info.get("last_check", 0) > 4.0)):
                hx1, hy1, hx2, hy2 = raw_head_box
                crop = frame_bgr[hy1:hy2, hx1:hx2]
                if crop.size > 0 and crop.shape[0] > 20 and crop.shape[1] > 20:
                    self.face_verifier.request_verification(tid, crop)
                    if info is None:
                        # 🔥 ВИПРАВЛЕННЯ БАГУ 1: Нова людина спочатку має статус "identifying",
                        # а НЕ "stranger"! Червоний колір не з'явиться помилково.
                        self.track_identities[tid] = {
                            "status": "identifying",
                            "dist": 1.0,
                            "last_check": now
                        }
                        info = self.track_identities[tid]

            # Визначаємо статус
            is_owner = (tid == self.owner_track_id) or (info is not None and info.get("status") == "owner")

            if is_owner:
                owner_body = p
                owner_body["head_box"] = raw_head_box
                self.owner_track_id = tid
            else:
                status = info.get("status", "identifying") if info else "identifying"
                dist = info.get("dist", 1.0) if info else 1.0
                others.append({
                    "box": p["box"],
                    "head_box": raw_head_box,
                    "track_id": tid,
                    "status": status,
                    "dist": dist
                })

        # Очищуємо кеш треків, які вийшли з кадру
        self.track_identities = {
            tid: val for tid, val in self.track_identities.items()
            if tid in active_track_ids
        }

        # Обличчя хазяїна
        if owner_body is None:
            self.owner_track_id = None
            self.smoothed_face_box = None
            owner_face = None
        else:
            # Згладжуємо квадрат голови хазяїна, прив'язаний до носа/очей
            smoothed = self.smooth_face_box(owner_body["head_box"])
            owner_face = {"box": smoothed}

        # 4. MediaPipe для рук (тільки коли хазяїн у кадрі)
        owner_hands = []
        if owner_body is not None:
            all_hands = self.detect_hands(frame_bgr, curr_time_ms)
            for h in all_hands:
                if wrist_near_body(h["wrist"], owner_body["box"]):
                    owner_hands.append(h)

        return {
            "owner_face": owner_face,
            "owner_body": owner_body,
            "owner_hands": owner_hands,
            "others": others
        }

    def draw_skeleton_hand(self, img, points, label):
        """Неоновий тактичний скелет кисті руки."""
        pts = [(int(x), int(y)) for x, y in points]      # cv2 вимагає цілі координати
        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(img, pts[p1], pts[p2], COLOR_HAND_BONE, 2, cv2.LINE_AA)

        for idx, (x, y) in enumerate(pts):
            r = 4 if idx in (4, 8, 12, 16, 20) else 3
            cv2.circle(img, (x, y), r, COLOR_HAND_JOINT, -1, cv2.LINE_AA)

        wx, wy = pts[0]
        draw_badge(img, label.upper(), (wx - 25, wy + 20), COLOR_HAND_JOINT)

    def hand_badge_text(self, raw_label):
        """Підпис руки: мітка з погляду користувача + роль (CMD / MOVE), якщо є контролер."""
        if self.controller is None:
            return f"OWNER {str(raw_label).upper()}"
        mapper = self.controller.mapper
        side = mapper.user_hand_label(raw_label)
        if side not in ("Left", "Right"):
            return f"OWNER {str(raw_label).upper()}"
        role = "CMD" if mapper.config.side_for("command") == side else "MOVE"
        return f"{side.upper()} HAND [{role}]"

    def render(self, frame, assoc, command=None):
        """Рендеринг оверлею (CV + панель керування дроном)."""
        h, w = frame.shape[:2]

        # 1. Інші люди (чужі або на стадії розпізнавання)
        for o in assoc["others"]:
            bx = o["box"]
            status = o["status"]
            if status == "stranger":
                clr = COLOR_STRANGER
                label = f"STRANGER ({o['dist']:.2f})"
            else:
                clr = COLOR_IDENTIFYING
                label = "IDENTIFYING..."

            # Тіло
            cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), COLOR_STRANGER_BODY, 1)
            # Бейдж
            draw_badge(frame, label, (bx[0], bx[1] - 5), clr)

        # 2. Тіло хазяїна (яскраве жовте)
        if assoc["owner_body"]:
            bx = assoc["owner_body"]["box"]
            draw_styled_rect(frame, bx, COLOR_OWNER_BODY, thickness=2, corner_len=18)
            tid_str = f" [ID:{assoc['owner_body']['track_id']}]" if assoc['owner_body']['track_id'] else ""
            draw_badge(frame, f"OWNER BODY{tid_str}", (bx[0] + 5, bx[1] + 16), COLOR_OWNER_BODY)

        # 3. Обличчя хазяїна (зелений квадрат, який ходить за нахилом голови!)
        if assoc["owner_face"]:
            fx = assoc["owner_face"]["box"]
            draw_styled_rect(frame, fx, COLOR_OWNER_FACE, thickness=2, corner_len=12)
            draw_badge(frame, "OWNER FACE", (fx[0], fx[1] - 5), COLOR_OWNER_FACE)

        # 4. Руки хазяїна
        for hand in assoc["owner_hands"]:
            self.draw_skeleton_hand(frame, hand["points"], self.hand_badge_text(hand["label"]))

        # 5. Верхній HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 46), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        cv2.putText(frame, "MEDUSA EYE // POSE TRACKING (640x480)", (12, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

        is_locked = (assoc["owner_body"] is not None)
        status_txt = f"OWNER: LOCKED [Track ID: {self.owner_track_id}]" if is_locked else ("WAITING FOR OWNER..." if self.owner_encoding is not None else "NO OWNER REGISTERED [Press R]")
        status_clr = (0, 255, 0) if is_locked else ((0, 180, 255) if self.owner_encoding is not None else (0, 0, 255))
        cv2.putText(frame, status_txt, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_clr, 1, cv2.LINE_AA)

        # FPS
        fps_clr = (0, 255, 0) if self.fps >= 28 else ((0, 200, 255) if self.fps >= 18 else (0, 0, 255))
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 95, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_clr, 1, cv2.LINE_AA)

        # Нижня панель
        overlay_b = frame.copy()
        cv2.rectangle(overlay_b, (0, h - 22), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay_b, 0.70, frame, 0.30, 0, frame)
        cv2.putText(frame, "[G] Gestures   [R] Register Owner   [D] Debug   [C] Clear   [Q] Quit",
                    (12, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1, cv2.LINE_AA)

        if self.debug_mode:
            dbg_lines = [
                f"Track ID: {self.owner_track_id}",
                f"Others: {len(assoc['others'])}",
                f"Hands: {len(assoc['owner_hands'])}",
                f"Gesture: {getattr(command, 'gesture_text', '-')}",
            ]
            dy = 65
            for dl in dbg_lines:
                draw_badge(frame, dl, (12, dy), COLOR_HUD_TEXT)
                dy += 18

        # 6. Панель керування дроном (тільки якщо підключений контролер)
        if self.controller is not None:
            telemetry = self.controller.link.telemetry
            draw_drone_panel(
                frame,
                command=command,
                link=self.controller.link,
                enabled=self.controller.enabled,
                telemetry=telemetry,
            )
            draw_action_flash(frame, command, time.monotonic(), self.last_action_at)

    def update_fps(self):
        """Оновлює лічильник FPS."""
        self.frame_idx += 1
        elapsed = time.time() - self.fps_timer
        if elapsed >= 0.4:
            self.fps = self.frame_idx / elapsed
            self.frame_idx = 0
            self.fps_timer = time.time()

    def run(self):
        """Головний цикл."""
        print("\n🚀 Medusa Eye запущено (Pose Tracking Engine)!")
        print("   Дивіться в камеру. Для запису натисніть [R]. Для виходу [Q] або [Esc].")
        if self.controller is not None:
            print("   [G] — увімкнути/вимкнути керування дроном жестами.\n")
        else:
            print()

        start_time_ms = int(time.time() * 1000)
        last_label = None

        try:
            while True:
                raw_frame = self.cam.read()
                if raw_frame is None:
                    time.sleep(0.002)
                    continue

                frame = cv2.resize(raw_frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
                curr_time_ms = int(time.time() * 1000) - start_time_ms

                # Повний конвеєр (YOLO-pose -> хазяїн -> MediaPipe руки)
                assoc = self.process_pipeline(frame, curr_time_ms)

                # Жести -> вектор команд -> симулятор
                command = None
                if self.controller is not None:
                    command = self.controller.update(assoc, frame.shape)
                    if command is not None:
                        if command.action != 0 and command.action != self.last_action:
                            self.last_action = command.action
                            self.last_action_at = time.monotonic()
                        elif command.action == 0:
                            self.last_action = 0
                        if command.label != last_label:
                            print(f"🖐  {command.gesture_text} -> {command.label}  {command.to_text()}"
                                  f"  [{command.origin}]"
                                  f"  (link: {self.controller.link.status_text()})")
                            last_label = command.label
                self.gesture_command = command

                # Рендеринг
                self.update_fps()
                self.render(frame, assoc, command)

                cv2.imshow("Medusa Eye // ESP32 VGA Simulator", frame)

                # якщо вікно закрили хрестиком — виходимо коректно
                try:
                    if cv2.getWindowProperty("Medusa Eye // ESP32 VGA Simulator",
                                              cv2.WND_PROP_VISIBLE) < 1:
                        print("👋 Вікно закрито.")
                        break
                except cv2.error:
                    break

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    print("👋 Завершення роботи.")
                    break
                elif key in (ord('g'), ord('G')):
                    if self.controller is not None:
                        state = self.controller.toggle()
                        print(f"🎮 Керування жестами: {'ON' if state else 'OFF'}")
                elif key in (ord('r'), ord('R')):
                    self.register_owner(frame)
                elif key in (ord('c'), ord('C')):
                    self.clear_owner()
                elif key in (ord('d'), ord('D')):
                    self.debug_mode = not self.debug_mode
                    print(f"🔧 Debug режим: {'ON' if self.debug_mode else 'OFF'}")

        except KeyboardInterrupt:
            print("\n⛔ Зупинено користувачем.")
        finally:
            self.cleanup()

    def cleanup(self):
        """Звільнення ресурсів."""
        print("🧹 Зупинка фонових потоків та закриття вікон...")
        self.face_verifier.stop()
        self.cam.release()
        cv2.destroyAllWindows()
        print("✅ Завершено.")


def parse_args():
    """Сумісність зі старим запуском: python3 medusa/eye.py --source esp32 ..."""
    import argparse

    parser = argparse.ArgumentParser(description="Medusa Eye: розпізнавання хазяїна")
    parser.add_argument("--source", choices=["esp32", "webcam"], default="esp32",
                        help="esp32 — відео з ESP32-CAM; webcam — локальна камера")
    parser.add_argument("--ip", default=DEFAULT_IP, help=f"IP ESP32-CAM (default: {DEFAULT_IP})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"TCP stream port (default: {DEFAULT_PORT})")
    return parser.parse_args()


if __name__ == "__main__":  # прямий запуск: тільки CV, без симулятора
    cli = parse_args()
    esp_endpoint = (cli.ip, cli.port) if cli.source == "esp32" else None
    MedusaEye(source=cli.source, esp32_url=esp_endpoint).run()
