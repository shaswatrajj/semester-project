"""Extract three-region RGB traces and POS rPPG from a face video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from localphysformer.config import SIGNALS_DIR
from localphysformer.pipeline import extract_video_rppg


def main() -> None:
    parser = argparse.ArgumentParser(description="LocalPhysFormer Phase 2: extract POS rPPG")
    parser.add_argument("--video", required=True, help="Path to a face video")
    parser.add_argument(
        "--out",
        default="",
        help="Output .npz path (default: data/signals/<video_stem>_rppg.npz)",
    )
    parser.add_argument("--no-bandpass", action="store_true", help="Skip heart-rate bandpass")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out) if args.out else SIGNALS_DIR / f"{video_path.stem}_rppg.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rgb_traces, rppg, fps = extract_video_rppg(
        video_path,
        apply_bandpass=not args.no_bandpass,
    )

    payload = {"fps": fps, "regions": np.array(list(rgb_traces.keys()))}
    for name, rgb in rgb_traces.items():
        payload[f"rgb_{name}"] = rgb
        payload[f"rppg_{name}"] = rppg[name]
        n_valid = int(np.isfinite(rgb).all(axis=1).sum())
        print(f"{name}: {rgb.shape[0]} frames, {n_valid} with a detected face")

    np.savez_compressed(out_path, **payload)
    print(f"fps={fps:.3f}")
    print(f"saved {out_path}")
    print("Transformer classification is not implemented in Phase 2.")


if __name__ == "__main__":
    main()
