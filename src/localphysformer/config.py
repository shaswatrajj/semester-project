"""Project paths and signal-processing settings."""

from pathlib import Path

from localphysformer import ROI_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SIGNALS_DIR = DATA_DIR / "signals"
MODELS_DIR = PROJECT_ROOT / "models"

REGIONS = ROI_NAMES
DEFAULT_FPS = 30

# POS overlap-add window (Wang et al., 2017).
POS_WINDOW_SEC = 1.6

# Heart-rate band used after POS (about 42–150 BPM).
BANDPASS_HZ = (0.7, 2.5)

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"
