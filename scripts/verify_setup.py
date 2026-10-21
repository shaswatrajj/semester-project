"""Confirm Phase 1 package imports and dependency versions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import mediapipe as mp
import numpy as np
import scipy

import localphysformer
from localphysformer.config import DEFAULT_FPS, REGIONS, SIGNALS_DIR
from localphysformer.pipeline import describe_pipeline


def main() -> None:
    print("LocalPhysFormer setup check")
    print(f"  package version: {localphysformer.__version__}")
    print(f"  ROI names: {', '.join(localphysformer.ROI_NAMES)}")
    print(f"  config regions: {', '.join(REGIONS)}")
    print(f"  default FPS: {DEFAULT_FPS}")
    print(f"  signals dir: {SIGNALS_DIR}")
    print()
    print(describe_pipeline())
    print()
    print("Dependency versions:")
    print(f"  numpy: {np.__version__}")
    print(f"  opencv: {cv2.__version__}")
    print(f"  scipy: {scipy.__version__}")
    print(f"  mediapipe: {mp.__version__}")
    print()
    print("Setup OK. Phase 2 implements ROI + POS. Transformer is not implemented yet.")


if __name__ == "__main__":
    main()
