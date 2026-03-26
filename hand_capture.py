import os
import cv2


SAVE_DIR = "C:/Users/admin/hand_dataset"
ROI_SIZE = 300


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    person_name = input("Enter label/name for this hand data: ").strip()
    if not person_name:
        print("Name is required.")
        return

    out_dir = os.path.join(SAVE_DIR, person_name)
    ensure_dir(out_dir)

    existing = [f for f in os.listdir(out_dir) if f.lower().endswith(".jpg")]
    count = len(existing)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Move your hand inside the square. Press S to save, Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        x1 = (w - ROI_SIZE) // 2
        y1 = (h - ROI_SIZE) // 2
        x2 = x1 + ROI_SIZE
        y2 = y1 + ROI_SIZE

        roi = frame[y1:y2, x1:x2]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{person_name} samples: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Put hand in square | S=save | Q=quit",
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Hand Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            count += 1
            save_path = os.path.join(out_dir, f"{count:04d}.jpg")
            cv2.imwrite(save_path, roi)
            print(f"Saved: {save_path}")

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
