
import subprocess
import sys
import urllib.request
import os

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_NAME = "hand_landmarker.task"

print("=" * 50)
print("  OpenCV Course — إعداد البيئة")
print("=" * 50)

# ── 1. تثبيت المكتبات
print("\n[1/2] تثبيت المكتبات ...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
    "opencv-python",
    "opencv-contrib-python",
    "mediapipe==0.10.30",
    "numpy",
])
print("     تم تثبيت المكتبات بنجاح ✓")

# ── 2. تحميل نموذج اليد
if os.path.isfile(MODEL_NAME):
    print(f"\n[2/2] النموذج موجود بالفعل: {MODEL_NAME} ✓")
else:
    print(f"\n[2/2] تحميل نموذج اليد ({MODEL_NAME}) ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)
    print("     تم تحميل النموذج بنجاح ✓")

print("\n" + "=" * 50)
print("  الإعداد اكتمل! يمكنك الآن تشغيل أي ملف.")
print("  مثال: python hand_distance_x.py")
print("=" * 50)
