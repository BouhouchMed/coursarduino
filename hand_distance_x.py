import ctypes

# Fix: mediapipe shared lib does not export 'free' on Windows (Python 3.13)
# We intercept CDLL.__getitem__ to redirect 'free' to ucrtbase.dll
_orig_cdll_getitem = ctypes.CDLL.__getitem__

def _patched_cdll_getitem(self, name_or_ordinal):
    try:
        return _orig_cdll_getitem(self, name_or_ordinal)
    except AttributeError:
        if name_or_ordinal == "free":
            return _orig_cdll_getitem(ctypes.CDLL("ucrtbase.dll"), "free")
        raise

ctypes.CDLL.__getitem__ = _patched_cdll_getitem

import math
import cv2
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import mediapipe as mp


MODEL_PATH = "C:/Users/admin/hand_landmarker.task"

# ── نقاط المفاصل الـ 21
# 0: WRIST  1-4: THUMB  5-8: INDEX  9-12: MIDDLE  13-16: RING  17-20: PINKY
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# HAND_CONNECTIONS لرسم خطوط ربط المفاصل
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20),# pinky
    (5, 9), (9, 13), (13, 17),            # palm
]

# نطاق المسافة (بكسل) → x بين 0 و 100
MIN_DIST = 10.0
MAX_DIST = 200.0
MIN_X = 0.0
MAX_X = 100.0


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def scale_to_x(distance: float) -> float:
    distance = clamp(distance, MIN_DIST, MAX_DIST)
    return (distance - MIN_DIST) / (MAX_DIST - MIN_DIST) * (MAX_X - MIN_X) + MIN_X


def draw_landmarks(frame: np.ndarray, lms: list, h: int, w: int) -> None:
    for start, end in HAND_CONNECTIONS:
        x1, y1 = int(lms[start].x * w), int(lms[start].y * h)
        x2, y2 = int(lms[end].x * w), int(lms[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 0), 2)
    for idx, lm in enumerate(lms):
        cx, cy = int(lm.x * w), int(lm.y * h)
        color = (0, 0, 255) if idx in (4, 8) else (0, 255, 0)
        cv2.circle(frame, (cx, cy), 6 if idx in (4, 8) else 4, color, -1)




def main() -> None:
    options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    x = 0.0
    frame_ts = 0
    print("Running hand joint tracking. Press Q or ESC to exit.")

    with mp_vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            frame_ts += 33  # ~30 fps timestamp in ms

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_image, frame_ts)

            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                draw_landmarks(frame, lms, h, w)

                # ── حساب المسافة بين طرف الإبهام (4) وطرف السبابة (8)
                tx, ty = int(lms[4].x * w), int(lms[4].y * h)
                ix, iy = int(lms[8].x * w), int(lms[8].y * h)
                dist = math.hypot(ix - tx, iy - ty)
                x = scale_to_x(dist)

                # ── خط المسافة
                mid = ((tx + ix) // 2, (ty + iy) // 2)
                cv2.line(frame, (tx, ty), (ix, iy), (255, 100, 0), 2)
                cv2.putText(frame, f"{dist:.0f}px", mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

            # ── شريط تمثيل x
            bar_w = int((x / MAX_X) * (w - 40))
            cv2.rectangle(frame, (20, h - 50), (w - 20, h - 30), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, h - 50), (20 + bar_w, h - 30), (0, 220, 255), -1)

            # ── نصوص المعلومات
            cv2.putText(frame, f"x = {x:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "Thumb(4) <-> Index(8) distance",
                        (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, "Q / ESC = quit",
                        (w - 160, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Hand Joints -> x (0-100)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
