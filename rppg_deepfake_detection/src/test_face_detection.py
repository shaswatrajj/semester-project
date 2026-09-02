import cv2

print("Starting...")

from face_detection import create_detector, detect_faces
from roi_extraction import extract_face, resize_face

print("Creating detector...")

detector = create_detector()

print("Detector created!")

cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    faces = detect_faces(frame, detector)

    print("Faces detected:", len(faces), end="\r")

    for bbox in faces:
        x, y, w, h = bbox

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        face = extract_face(frame, bbox)

        if face is not None:
            face = resize_face(face)

    cv2.imshow("Face Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()

print("\nFinished.")