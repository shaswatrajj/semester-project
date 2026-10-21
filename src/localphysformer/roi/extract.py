"""Extract per-frame mean RGB traces from forehead and cheek ROIs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
)

from localphysformer import ROI_NAMES
from localphysformer.config import DEFAULT_FPS, FACE_LANDMARKER_PATH
from localphysformer.roi.landmarks import ROI_LANDMARKS
from localphysformer.roi.model import ensure_face_landmarker_model


def mean_rgb_in_polygon(frame_rgb: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """Average RGB over the convex hull of polygon vertices. NaN if empty."""
    height, width = frame_rgb.shape[:2]
    pts = np.asarray(points_xy, dtype=np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    if len(pts) < 3:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    hull = cv2.convexHull(pts)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    if int(np.count_nonzero(mask)) == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    mean_rgb = cv2.mean(frame_rgb, mask=mask)
    return np.array(mean_rgb[:3], dtype=np.float64)


def _pixel_landmarks(face_landmarks, width: int, height: int) -> np.ndarray:
    points = np.array([[lm.x * width, lm.y * height] for lm in face_landmarks], dtype=np.float64)
    return points


def create_face_landmarker(model_path: Path | None = None) -> FaceLandmarker:
    path = ensure_face_landmarker_model(model_path or FACE_LANDMARKER_PATH)
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(path)),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


def extract_rgb_traces(
    video_path: str | Path,
    landmarker: FaceLandmarker | None = None,
) -> tuple[dict[str, np.ndarray], float]:
    """Read a video and return {region: (T, 3) RGB} plus fps.

    Frames without a detected face store NaNs. Caller owns `landmarker` if
    one is passed; otherwise a landmarker is created and closed here.
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    owns_landmarker = landmarker is None
    if owns_landmarker:
        landmarker = create_face_landmarker()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        if owns_landmarker:
            landmarker.close()
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = float(DEFAULT_FPS)

    traces = {name: [] for name in ROI_NAMES}
    frame_index = 0
    last_timestamp_ms = -1
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            timestamp_ms = int(round(1000.0 * frame_index / fps))
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms
            mp_image = Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            if result.face_landmarks:
                pixels = _pixel_landmarks(result.face_landmarks[0], width, height)
                for name in ROI_NAMES:
                    indices = ROI_LANDMARKS[name]
                    points = pixels[list(indices)]
                    traces[name].append(mean_rgb_in_polygon(rgb, points))
            else:
                nan = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                for name in ROI_NAMES:
                    traces[name].append(nan)
            frame_index += 1
    finally:
        cap.release()
        if owns_landmarker:
            landmarker.close()

    stacked = {name: np.vstack(values) if values else np.empty((0, 3)) for name, values in traces.items()}
    return stacked, fps
