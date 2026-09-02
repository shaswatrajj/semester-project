import cv2
from face_detection import create_detector, detect_faces
from roi_extraction import extract_face, resize_face


def process_video(video_path, max_frames=None):
    detector = create_detector()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video.")
        detector.close()
        return []

    face_frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        faces = detect_faces(frame, detector)

        if faces:
            face = extract_face(frame, faces[0])

            if face is not None:
                face = resize_face(face)
                face_frames.append(face)

        frame_count += 1

        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()
    detector.close()

    return face_frames