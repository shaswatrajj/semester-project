import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "../models/face_detector.task"


def detect_faces(frame, detector):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    results = detector.detect(mp_image)

    faces = []

    for detection in results.detections:
        bbox = detection.bounding_box

        x = max(0, bbox.origin_x)
        y = max(0, bbox.origin_y)
        w = min(bbox.width, frame.shape[1] - x)
        h = min(bbox.height, frame.shape[0] - y)

        faces.append((x, y, w, h))

    return faces


def create_detector():
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )

    return vision.FaceDetector.create_from_options(options)