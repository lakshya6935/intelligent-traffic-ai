"""Central configuration. Every path and default lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "dataset" / "raw"
LABEL_FILE = RAW_DIR / "train-anomaly-results.txt"
CACHE_DIR = ROOT / "dataset" / "cache"
FEATURES_CSV = ROOT / "dataset" / "features.csv"
MODEL_PATH = ROOT / "models" / "accident_model.pkl"
OUTPUT_DIR = ROOT / "outputs"

# Detection / tracking
DEFAULT_YOLO = "yolo11n.pt"      # any Ultralytics detection weights work (yolov8n.pt, yolo26n.pt, ...)
VEHICLE_CLASSES = [2, 3, 5, 7]   # COCO: car, motorcycle, bus, truck
TRACKER = "bytetrack.yaml"
CONF = 0.25
IMGSZ = 640                      # raise to 960 for small/far vehicles (slower)
STRIDE = 3                       # process every 3rd frame (30 fps -> 10 fps)

# Windows used for feature extraction and classification
WIN_SEC = 10.0
HOP_SEC = 5.0
MIN_EVENT_WINDOWS = 2            # consecutive flagged windows needed to raise an alert
