from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FRAME_DIR = DATA_DIR / "frames"
DB_PATH = DATA_DIR / "smart_curb.db"

DEFAULT_MODEL = "yolov8n.pt"
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    FRAME_DIR.mkdir(exist_ok=True)
