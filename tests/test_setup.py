"""Phase 1 import and constant checks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import localphysformer
from localphysformer.config import REGIONS
from localphysformer.pipeline import PIPELINE_STAGES, describe_pipeline


class TestPhase1Setup(unittest.TestCase):
    def test_package_version(self) -> None:
        self.assertEqual(localphysformer.__version__, "0.2.0-phase2")

    def test_roi_names(self) -> None:
        self.assertEqual(
            localphysformer.ROI_NAMES,
            ("forehead", "left_cheek", "right_cheek"),
        )
        self.assertEqual(REGIONS, localphysformer.ROI_NAMES)

    def test_pipeline_description(self) -> None:
        text = describe_pipeline()
        self.assertIn("POS", text)
        self.assertIn("Transformer", text)
        self.assertEqual(len(PIPELINE_STAGES), 4)


if __name__ == "__main__":
    unittest.main()
