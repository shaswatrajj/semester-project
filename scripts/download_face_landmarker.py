"""Download the MediaPipe Face Landmarker model used for ROI extraction."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from localphysformer.roi.model import ensure_face_landmarker_model


def main() -> None:
    path = ensure_face_landmarker_model()
    print(f"Model ready: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
