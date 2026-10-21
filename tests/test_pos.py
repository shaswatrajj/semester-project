"""POS recovers a known pulsatile component from synthetic RGB."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from localphysformer.rppg.pos import interpolate_nans, pos_from_rgb, traces_to_rppg


class TestPOS(unittest.TestCase):
    def test_pos_correlates_with_synthetic_pulse(self) -> None:
        fps = 30.0
        seconds = 10
        time = np.arange(int(fps * seconds)) / fps
        pulse = np.sin(2 * np.pi * 1.2 * time)
        rgb = np.column_stack(
            [
                140.0 + 4.0 * pulse,
                90.0 + 18.0 * pulse,
                70.0 + 7.0 * pulse,
            ]
        )
        rgb += np.random.default_rng(0).normal(0, 0.2, size=rgb.shape)
        recovered = pos_from_rgb(rgb, fps)
        self.assertFalse(np.isnan(recovered).all())
        corr = np.corrcoef(pulse, recovered)[0, 1]
        self.assertGreater(abs(corr), 0.5)

    def test_all_nan_input(self) -> None:
        rgb = np.full((20, 3), np.nan)
        out = pos_from_rgb(rgb, fps=30)
        self.assertEqual(out.shape, (20,))
        self.assertTrue(np.isnan(out).all())

    def test_interpolate_and_traces_to_rppg(self) -> None:
        rgb = np.ones((10, 3), dtype=np.float64) * 100
        rgb[3] = np.nan
        filled = interpolate_nans(rgb)
        self.assertFalse(np.isnan(filled[3]).any())
        signals = traces_to_rppg({"forehead": filled}, fps=30.0, apply_bandpass=False)
        self.assertIn("forehead", signals)
        self.assertEqual(signals["forehead"].shape, (10,))


if __name__ == "__main__":
    unittest.main()
