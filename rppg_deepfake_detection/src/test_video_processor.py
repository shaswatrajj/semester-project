import cv2

from face_detection import create_detector, detect_faces
from roi_extraction import extract_face, resize_face


detector = create_detector()
cap = cv2.VideoCapture(0)

count = 0

while count < 30:
    ret, frame = cap.read()

    if not ret:
        break

    faces = detect_faces(frame, detector)

    if faces:
        face = extract_face(frame, faces[0])

        if face is not None:
            face = resize_face(face)
            print(f"Frame {count + 1}: Face extracted - {face.shape}")

    count += 1

cap.release()
detector.close()

print(f"Processed {count} frames.")