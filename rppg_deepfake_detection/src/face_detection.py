import cv2
import mediapipe as mp

video_path = "../data/videos/test.mp4"

cap = cv2.VideoCapture(video_path)

mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
) as face_detection:

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(rgb_frame)

        if results.detections:

            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box

                h, w, _ = frame.shape

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)

                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Keep coordinates inside the frame
                x = max(0, x)
                y = max(0, y)

                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Face Detection", frame)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()