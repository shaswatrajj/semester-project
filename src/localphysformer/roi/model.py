"""Download the MediaPipe Face Landmarker .task file if it is missing."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from localphysformer.config import FACE_LANDMARKER_PATH, FACE_LANDMARKER_URL


def ensure_face_landmarker_model(destination: Path | None = None) -> Path:
    """Return the model path, downloading it once if needed."""
    path = Path(destination or FACE_LANDMARKER_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        return path
    print(f"Downloading Face Landmarker model to {path} ...")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, path)
    return path
