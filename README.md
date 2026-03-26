# OpenCV & Hand Detection Project

مشروع تعليمي شامل لـ OpenCV يتضمن:
- كشف الوجه
- التعرف على الوجه بالاسم
- كشف وتتبع اليد
- معالجة الصور والفيديو

## الملفات الرئيسية

### 1. كشف الوجه
- `face_detection.py` - كشف الوجه من الكاميرا بدون تمييز (Haar Cascade)

### 2. التعرف على الوجه بالاسم
- `capture_face_samples.py` - جمع صور لشخص معين
- `train_face_model.py` - تدريب نموذج LBPH للتعرف
- `face_recognition_live.py` - التعرف على الوجه من الكاميرا بالاسم

### 3. كشف اليد
- `hand_capture.py` - تسجيل صور اليد
- `hand_distance_x.py` - حساب متغير x يتغير حسب المسافة بين الإبهام والسبابة

### 4. دروس
- `opencv_lesson_start.py` - مثال بداية بسيط

## المتطلبات

```bash
pip install opencv-python opencv-contrib-python mediapipe numpy
```

## التشغيل

```bash
# مثال: كشف الوجه
python face_detection.py

# مثال: التعرف على الوجه بالاسم
python capture_face_samples.py      # جمع صور
python train_face_model.py          # تدريب النموذج
python face_recognition_live.py     # التعرف

# مثال: كشف اليد والحصول على x
python hand_distance_x.py
```

## الملاحظات

- يتطلب كاميرا ويب
- جودة الإضاءة مهمة جدًا لدقة الكشف
- للتعرف على الوجه بدقة: اجمع 30-50 صورة لكل شخص بزوايا وإضاءة مختلفة

## المؤلف
OpenCV Learning Project

## الترخيص
MIT
