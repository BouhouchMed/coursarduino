import os
import cv2
import numpy as np


DATASET_DIR = "C:/Users/admin/face_dataset"
MODEL_PATH = "C:/Users/admin/face_model.yml"
LABELS_PATH = "C:/Users/admin/face_labels.txt"


def main() -> None:
    if not os.path.isdir(DATASET_DIR):
        print("Dataset directory not found. Run capture_face_samples.py first.")
        return

    if not hasattr(cv2, "face"):
        print("cv2.face module is not available. Install opencv-contrib-python.")
        return

    label_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    if not label_names:
        print("No person folders found in dataset.")
        return

    label_to_id = {name: idx for idx, name in enumerate(label_names)}

    images = []
    labels = []

    for name in label_names:
        person_dir = os.path.join(DATASET_DIR, name)
        for file_name in os.listdir(person_dir):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_dir, file_name)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue

            if gray.shape != (200, 200):
                gray = cv2.resize(gray, (200, 200))

            images.append(gray)
            labels.append(label_to_id[name])

    if len(images) < 5:
        print("Not enough training images. Capture more samples first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        for name, idx in label_to_id.items():
            f.write(f"{idx},{name}\n")

    print(f"Training complete. Images: {len(images)}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Labels saved to: {LABELS_PATH}")


if __name__ == "__main__":
    main()
