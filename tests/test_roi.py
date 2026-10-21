"""ROI averaging and landmark tables (no live face required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from localphysformer import ROI_NAMES
from localphysformer.roi.extract import mean_rgb_in_polygon
from localphysformer.roi.landmarks import ROI_LANDMARKS


class TestROI(unittest.TestCase):
    def test_landmark_tables(self) -> None:
        self.assertEqual(set(ROI_LANDMARKS), set(ROI_NAMES))
        for name, indices in ROI_LANDMARKS.items():
            self.assertGreaterEqual(len(indices), 3, msg=name)
            self.assertTrue(all(0 <= i < 468 for i in indices), msg=name)

    def test_mean_rgb_in_polygon(self) -> None:
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        frame[5:15, 5:15] = (10, 20, 30)
        points = np.array([[5, 5], [14, 5], [14, 14], [5, 14]], dtype=np.int32)
        mean = mean_rgb_in_polygon(frame, points)
        np.testing.assert_allclose(mean, [10, 20, 30], atol=1.0)

    def test_empty_polygon_is_nan(self) -> None:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        mean = mean_rgb_in_polygon(frame, np.array([[1, 1], [2, 2]], dtype=np.int32))
        self.assertTrue(np.isnan(mean).all())


if __name__ == "__main__":
    unittest.main()
