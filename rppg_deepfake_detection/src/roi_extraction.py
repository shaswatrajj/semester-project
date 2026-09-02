import cv2


def extract_face(frame, bbox):
    x, y, w, h = bbox

    x = max(0, x)
    y = max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)

    face = frame[y:y + h, x:x + w]

    if face.size == 0:
        return None

    return face


def resize_face(face, size=(224, 224)):
    return cv2.resize(face, size)
