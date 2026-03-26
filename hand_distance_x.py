import math
import cv2
import numpy as np


# Distance range in pixels (adjust based on your camera setup)
MIN_DIST = 20.0
MAX_DIST = 220.0

# Target x range
MIN_X = 0.0
MAX_X = 100.0


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def map_distance_to_x(distance: float) -> float:
    distance = clamp(distance, MIN_DIST, MAX_DIST)
    ratio = (distance - MIN_DIST) / (MAX_DIST - MIN_DIST)
    return MIN_X + ratio * (MAX_X - MIN_X)


def main() -> None:
    """
    Hand tracking using skin color detection and contours.
    Detects hand size and maps it to x variable (0-100).
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    x = 0.0
    print("Running hand tracking (contour-based). Press Q or ESC to exit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Skin color range - lower range (reddish)
        lower1 = np.array([0, 20, 70], dtype=np.uint8)
        upper1 = np.array([20, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        
        # Skin color range - upper range (reddish)
        lower2 = np.array([170, 20, 70], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphology operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (expected to be the hand)
            hand = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(hand)
            
            if area > 500:  # Minimum hand size threshold
                x_hand, y_hand, w_hand, h_hand = cv2.boundingRect(hand)
                cv2.rectangle(frame, (x_hand, y_hand), (x_hand + w_hand, y_hand + h_hand), (0, 255, 0), 2)
                
                # Calculate hand size as distance
                distance = math.sqrt(w_hand**2 + h_hand**2)
                x = map_distance_to_x(distance)
                
                cv2.putText(
                    frame,
                    f"Hand size: {distance:.1f}px",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.putText(
            frame,
            f"x: {x:.2f}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Move hand closer/further to change x",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Hand Distance -> x", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
