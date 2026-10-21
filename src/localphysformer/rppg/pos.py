"""POS (Plane-Orthogonal-to-Skin) rPPG from Wang et al., 2017."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, detrend, filtfilt

from localphysformer.config import BANDPASS_HZ, POS_WINDOW_SEC

# Projection onto the plane orthogonal to the skin-tone axis [1, 1, 1].
_POS_PROJECTION = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float64)


def pos_from_rgb(
    rgb: np.ndarray,
    fps: float,
    window_sec: float = POS_WINDOW_SEC,
) -> np.ndarray:
    """Convert a single-region RGB trace to a 1-D POS rPPG signal.

    Args:
        rgb: Array of shape (T, 3) in RGB order, typically 0-255.
        fps: Video sampling rate.
        window_sec: Overlap-add window length in seconds (paper default 1.6).

    Returns:
        Array of shape (T,) with the overlap-added POS signal. All-NaN input
        returns all-NaN output.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape (T, 3), got {rgb.shape}")
    if fps <= 0:
        raise ValueError("fps must be positive")

    length = rgb.shape[0]
    if length == 0 or np.isnan(rgb).all():
        return np.full(length, np.nan, dtype=np.float64)

    window = max(int(round(fps * window_sec)), 2)
    if window > length:
        window = length

    output = np.zeros(length, dtype=np.float64)
    for start in range(0, length - window + 1):
        chunk = rgb[start : start + window]
        if np.isnan(chunk).any():
            continue
        means = np.mean(chunk, axis=0)
        if np.any(means == 0):
            continue
        normalized = (chunk / means).T
        projected = _POS_PROJECTION @ normalized
        std_s0 = float(np.std(projected[0]))
        std_s1 = float(np.std(projected[1]))
        if std_s1 == 0:
            continue
        pulse = projected[0] + (std_s0 / std_s1) * projected[1]
        pulse = pulse - np.mean(pulse)
        output[start : start + window] += pulse

    if np.allclose(output, 0.0):
        return np.full(length, np.nan, dtype=np.float64)
    return output


def bandpass_filter(
    signal: np.ndarray,
    fps: float,
    low_hz: float = BANDPASS_HZ[0],
    high_hz: float = BANDPASS_HZ[1],
    order: int = 2,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass. Skips filtering if the clip is too short."""
    signal = np.asarray(signal, dtype=np.float64)
    if np.isnan(signal).all() or signal.size < 12:
        return signal
    nyquist = fps / 2.0
    if not (0 < low_hz < high_hz < nyquist):
        return signal
    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")
    padlen = 3 * max(len(a), len(b))
    if signal.size <= padlen:
        return signal
    return filtfilt(b, a, detrend(signal, type="linear"))


def interpolate_nans(rgb: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs along time for each RGB channel."""
    out = np.asarray(rgb, dtype=np.float64).copy()
    if out.ndim != 2:
        raise ValueError("rgb must be (T, 3)")
    time = np.arange(out.shape[0])
    for channel in range(out.shape[1]):
        column = out[:, channel]
        missing = np.isnan(column)
        if missing.all() or not missing.any():
            continue
        column[missing] = np.interp(time[missing], time[~missing], column[~missing])
        out[:, channel] = column
    return out


def traces_to_rppg(
    rgb_traces: dict[str, np.ndarray],
    fps: float,
    apply_bandpass: bool = True,
) -> dict[str, np.ndarray]:
    """Run POS (and optional bandpass) on each named RGB region."""
    signals: dict[str, np.ndarray] = {}
    for name, rgb in rgb_traces.items():
        filled = interpolate_nans(rgb)
        pulse = pos_from_rgb(filled, fps)
        if apply_bandpass and not np.isnan(pulse).all():
            pulse = bandpass_filter(pulse, fps)
        signals[name] = pulse
    return signals
