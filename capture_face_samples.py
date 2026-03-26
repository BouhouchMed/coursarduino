import os
import cv2


DATASET_DIR = "C:/Users/admin/face_dataset"


def main() -> None:
    person_name = input("Enter person name (e.g., Ahmed): ").strip()
    if not person_name:
        print("Name is required.")
        return

    person_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("Could not load face cascade.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Press S to save face crop, Q to finish.")
    count = len([f for f in os.listdir(person_dir) if f.lower().endswith(".jpg")])

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"{person_name} samples: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "S=save, Q=quit", (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Capture Face Samples", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if len(faces) == 0:
                print("No face found in current frame.")
                continue

            # Save the largest detected face for better quality
            x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
            face_roi = gray[y : y + h, x : x + w]
            face_roi = cv2.resize(face_roi, (200, 200))

            count += 1
            file_path = os.path.join(person_dir, f"{count:04d}.jpg")
            cv2.imwrite(file_path, face_roi)
            print(f"Saved: {file_path}")

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
