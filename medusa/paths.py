"""Єдині шляхи проєкту (моделі, біометричний профіль, симулятор).

Усі шляхи рахуються від кореня репозиторію, тому скрипти можна запускати
з будь-якої робочої директорії:

    Drone/
    ├── medusa/            <- цей пакет
    ├── sim/               <- WebGL-симулятор дрона (server.py + index.html)
    ├── models/            <- hand_landmarker.task, yolov8n-pose.pt (качаються самі)
    └── data/              <- owner_face.npy (біометрія хазяїна, у git не потрапляє)
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MEDUSA_DIR = PROJECT_ROOT / "medusa"
SIM_DIR = PROJECT_ROOT / "sim"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

SIM_SERVER = SIM_DIR / "server.py"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
YOLO_POSE_PATH = MODELS_DIR / "yolov8n-pose.pt"
OWNER_FACE_PATH = DATA_DIR / "owner_face.npy"


def ensure_project_dirs() -> None:
    """Створює models/ та data/ (потрібно після свіжого клону репозиторію)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
