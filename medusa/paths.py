"""Unified project paths (models, biometric profile, simulator).

All paths are relative to the repository root, so scripts can be run
from any working directory:

    Drone/
    ├── medusa/            <- this package
    ├── sim/               <- WebGL drone simulator (server.py + index.html)
    ├── models/            <- hand_landmarker.task, yolov8n-pose.pt (downloaded automatically)
    └── data/              <- owner_face.npy (owner biometrics, not tracked in git)
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
ENV_FILE_PATH = PROJECT_ROOT / ".env"          # ELEVENLABS_API_KEY (not tracked in git)


def ensure_project_dirs() -> None:
    """Creates models/ and data/ directories (needed after a fresh clone)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
