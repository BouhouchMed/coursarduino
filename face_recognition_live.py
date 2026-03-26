import os
import cv2


MODEL_PATH = "C:/Users/admin/face_model.yml"
LABELS_PATH = "C:/Users/admin/face_labels.txt"
CONFIDENCE_THRESHOLD = 75.0


def load_labels(labels_path: str) -> dict[int, str]:
    labels: dict[int, str] = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, name = line.split(",", 1)
            labels[int(idx_str)] = name
    return labels


def main() -> None:
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(LABELS_PATH):
        print("Model/labels not found. Run train_face_model.py first.")
        return

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Could not load face cascade.")
        return

    if not hasattr(cv2, "face"):
        print("cv2.face module is not available. Install opencv-contrib-python.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    labels = load_labels(LABELS_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Face recognition running. Press Q or ESC to exit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for x, y, w, h in faces:
            face_roi = gray[y : y + h, x : x + w]
            face_roi = cv2.resize(face_roi, (200, 200))

            label_id, confidence = recognizer.predict(face_roi)
            if confidence <= CONFIDENCE_THRESHOLD:
                name = labels.get(label_id, "Unknown")
                text = f"{name} ({confidence:.1f})"
                color = (0, 255, 0)
            else:
                text = f"Unknown ({confidence:.1f})"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.putText(frame, "Press Q or ESC to quit", (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Face Recognition (By Name)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
