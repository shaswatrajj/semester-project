"""LocalPhysFormer pipeline: ROI RGB traces -> POS rPPG -> (later) Transformer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from localphysformer import ROI_NAMES
from localphysformer.roi.extract import extract_rgb_traces
from localphysformer.rppg.pos import traces_to_rppg

PIPELINE_STAGES = (
    "Load face video",
    "Extract RGB traces from forehead, left cheek, and right cheek",
    "Convert traces to rPPG with the POS algorithm",
    "Classify real vs fake with a temporal Transformer",
)


def describe_pipeline() -> str:
    """Return a human-readable description of the pipeline."""
    lines = ["LocalPhysFormer pipeline (Phase 2: ROI + POS):"]
    for index, stage in enumerate(PIPELINE_STAGES, start=1):
        marker = "done" if index <= 3 else "later"
        lines.append(f"  {index}. {stage} [{marker}]")
    lines.append(f"Regions: {', '.join(ROI_NAMES)}")
    return "\n".join(lines)


def extract_roi_traces(video_path: str | Path):
    """Landmark-based RGB traces for the three facial regions."""
    return extract_rgb_traces(video_path)


def extract_rppg_pos(
    rgb_traces: dict[str, np.ndarray],
    fps: float,
    apply_bandpass: bool = True,
) -> dict[str, np.ndarray]:
    """POS rPPG conversion for each region."""
    return traces_to_rppg(rgb_traces, fps, apply_bandpass=apply_bandpass)


def extract_video_rppg(
    video_path: str | Path,
    apply_bandpass: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """Run ROI extraction and POS on a video. Returns rgb, rppg, fps."""
    rgb_traces, fps = extract_roi_traces(video_path)
    rppg = extract_rppg_pos(rgb_traces, fps, apply_bandpass=apply_bandpass)
    return rgb_traces, rppg, fps


def classify_real_fake(_rppg_signals):
    """Temporal Transformer classifier. Implemented in a later phase."""
    raise NotImplementedError("The Transformer model is not part of Phase 2.")
