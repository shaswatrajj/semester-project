import cv2
import numpy as np

if __package__:
    from .face_detection import create_detector, detect_faces
    from .roi_extraction import extract_face, resize_face
else:
    # Support running this module directly from the src directory as well.
    from face_detection import create_detector, detect_faces
    from roi_extraction import extract_face, resize_face


def analyze_video(video_path, max_frames=None):
    detector = create_detector()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        detector.close()
        raise ValueError("Could not open the uploaded video.")

    face_frames = []
    frame_count = 0
    detected_count = 0
    brightness_values = []
    blur_values = []
    face_sizes = []
    motion_values = []
    previous_center = None
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            faces = detect_faces(frame, detector)
            frame_count += 1

            if faces:
                # Use the largest face to avoid analyzing a distant background face.
                bbox = max(faces, key=lambda item: item[2] * item[3])
                x, y, w, h = bbox
                face = extract_face(frame, bbox)

                if face is not None:
                    detected_count += 1
                    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    brightness_values.append(float(np.mean(gray)))
                    blur_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                    face_sizes.append((w, h))
                    center = (x + w / 2, y + h / 2)
                    if previous_center is not None:
                        diagonal = max((video_width ** 2 + video_height ** 2) ** 0.5, 1)
                        motion_values.append(float(np.hypot(center[0] - previous_center[0], center[1] - previous_center[1]) / diagonal))
                    previous_center = center
                    face_frames.append(resize_face(face))

            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        cap.release()
        detector.close()

    detection_rate = detected_count / frame_count if frame_count else 0.0
    avg_brightness = float(np.mean(brightness_values)) if brightness_values else None
    avg_blur = float(np.mean(blur_values)) if blur_values else None
    avg_face_width = float(np.mean([size[0] for size in face_sizes])) if face_sizes else None
    avg_face_height = float(np.mean([size[1] for size in face_sizes])) if face_sizes else None
    avg_motion = float(np.mean(motion_values)) if motion_values else 0.0

    checks = {
        "Face visibility": {
            "value": f"{detection_rate:.0%} of sampled frames",
            "status": "Good" if detection_rate >= 0.8 else "Needs attention" if detection_rate >= 0.5 else "Poor",
            "tip": "Keep your face in view throughout the clip." if detection_rate < 0.8 else None,
        },
        "Lighting": {
            "value": f"Average face brightness {avg_brightness:.0f}/255" if avg_brightness is not None else "No face detected",
            "status": "Not available" if avg_brightness is None else "Good" if 60 <= avg_brightness <= 205 else "Needs attention",
            "tip": "Use even, front-facing light and avoid strong backlighting." if avg_brightness is not None and not 60 <= avg_brightness <= 205 else None,
        },
        "Blur": {
            "value": f"Sharpness score {avg_blur:.0f}" if avg_blur is not None else "No face detected",
            "status": "Not available" if avg_blur is None else "Good" if avg_blur >= 60 else "Needs attention",
            "tip": "Hold still and keep the camera focused on your face." if avg_blur is not None and avg_blur < 60 else None,
        },
        "Resolution": {
            "value": f"Video {video_width}×{video_height}; face about {avg_face_width:.0f}×{avg_face_height:.0f}px" if avg_face_width is not None else f"Video {video_width}×{video_height}; no face detected",
            "status": "Not available" if avg_face_width is None else "Good" if video_width >= 320 and video_height >= 240 and min(avg_face_width, avg_face_height) >= 80 else "Needs attention",
            "tip": "Use a higher-resolution video and move closer so your face is larger in frame." if avg_face_width is not None and (video_width < 320 or video_height < 240 or min(avg_face_width, avg_face_height) < 80) else None,
        },
        "Motion": {
            "value": f"Average face movement {avg_motion:.1%} of frame diagonal per sample" if len(motion_values) else "Not enough face samples to estimate",
            "status": "Not available" if not motion_values else "Good" if avg_motion <= 0.04 else "Needs attention",
            "tip": "Keep your head steady and position the camera on a stable surface." if motion_values and avg_motion > 0.04 else None,
        },
    }
    overall_status = "Ready for analysis" if all(item["status"] == "Good" for item in checks.values()) else "May be unreliable"

    return {
        "face_frames": face_frames,
        "checks": checks,
        "overall_status": overall_status,
        "frames_analyzed": frame_count,
        "faces_found": detected_count,
    }


def process_video(video_path, max_frames=None):
    """Backward-compatible helper returning only extracted face frames."""
    return analyze_video(video_path, max_frames=max_frames)["face_frames"]
