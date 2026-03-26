import cv2
import numpy as np


def main() -> None:
    # Create a black canvas (height=480, width=640)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw a rectangle, a circle, and lesson title text
    cv2.rectangle(canvas, (40, 40), (300, 220), (0, 255, 0), 3)
    cv2.circle(canvas, (420, 140), 80, (255, 0, 0), 3)
    cv2.putText(
        canvas,
        "OpenCV Lesson 1",
        (130, 320),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Press Q to quit",
        (190, 380),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Lesson Start", canvas)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
