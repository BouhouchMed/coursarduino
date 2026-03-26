# OpenCV & Hand Detection Project

مشروع تعليمي شامل لـ OpenCV يتضمن كشف الوجه، التعرف بالاسم، وتتبع مفاصل اليد.

---

## 🚀 خطوات الطالب (أول مرة فقط)

### الخطوة 1 — تحميل Python
حمّل Python 3.12 أو أقل (يُنصح بـ 3.11) من:
> https://www.python.org/downloads/

> ⚠️ عند التثبيت: ضع علامة ✓ على **"Add Python to PATH"**

---

### الخطوة 2 — تحميل الملفات
```bash
git clone https://github.com/BouhouchMed/coursarduino.git
cd coursarduino
```
أو حمّل ZIP مباشرة من GitHub ثم افتح المجلد.

---

### الخطوة 3 — إعداد البيئة (مرة واحدة فقط)
```bash
python setup.py
```
سيقوم هذا الأمر بـ:
- تثبيت جميع المكتبات تلقائياً
- تحميل نموذج اليد

---

### الخطوة 4 — تشغيل الدروس

| الملف | الوصف |
|---|---|
| `opencv_lesson_start.py` | درس البداية — رسم الأشكال |
| `face_detection.py` | كشف الوجه من الكاميرا |
| `capture_face_samples.py` | جمع صور وجه لشخص معين |
| `train_face_model.py` | تدريب نموذج التعرف |
| `face_recognition_live.py` | التعرف على الوجه بالاسم |
| `hand_capture.py` | تصوير اليد وحفظها |
| `hand_distance_x.py` | تتبع مفاصل اليد → متغير x (0-100) |

```bash
python opencv_lesson_start.py
python face_detection.py
python hand_distance_x.py
```

---

## ⚠️ ملاحظات مهمة

- تأكد أن الكاميرا متصلة ومفتوحة
- `hand_distance_x.py` يحتاج ملف `hand_landmarker.task` (يُحمَّل تلقائياً عبر `setup.py`)
- الإضاءة الجيدة تُحسّن دقة الكشف
- اضغط **Q** أو **ESC** للخروج من أي برنامج

---

## المؤلف
bouhouch mohamed

## الترخيص
MIT
