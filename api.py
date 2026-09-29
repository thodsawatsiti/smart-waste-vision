"""
api.py
======
FastAPI server for Garbage Classification

Usage:
    python api.py

Endpoints:
    POST /predict  - Upload image, get classification result
    GET  /health   - Health check
    GET  /docs     - Swagger UI
"""

import io
import os
import time
import uuid
import platform
from datetime import datetime
import psutil
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from database import init_db, get_db, save_prediction, get_predictions, get_stats, update_feedback

# ====================================
# Config
# ====================================

# โมเดลที่เทรนบนภาพต้นฉบับ (ไม่ผ่าน image processing) + fine-tune ด้วย hard examples
# จากผลการทดลอง A/B: no_processing (96.14%) ดีกว่า with_processing (93.16%)
# fine-tune เพิ่มด้วยรูปถือมือจริง (feedback loop) แก้ปัญหา pose gap:
#   val set หลัก 95.8% (ลดจาก 96.14% เล็กน้อย)
#   แต่รูปถือมือจริงที่ไม่เคยเห็นตอนเทรน (holdout) จาก 37.5% -> 100% (16/16)
# โมเดลโปรดักชัน: no_processing + fine-tune ด้วย hard examples จาก feedback loop
#   val 95.8%  |  ภาพถ่ายจริงที่ไม่เคยเห็น (holdout) 37.5% -> 100%
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
CONF_THRESHOLD = 0.25

# โฟลเดอร์เก็บรูปที่ user upload — สำหรับ feedback loop + retrain
# หมายเหตุ: บน Hugging Face Spaces free tier โฟลเดอร์นี้ไม่ persistent (หายเมื่อ restart)
#          ถ้าต้องการเก็บถาวรต้องต่อ Storage Bucket หรือ external storage
UPLOAD_DIR = "uploads"
ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".webp")

# พักรูปที่ทำนายแล้วไว้ในหน่วยความจำ (key = prediction_id)
# จะเซฟลง uploads/ ก็ต่อเมื่อ user กด feedback (confirm) เท่านั้น
pending_images = {}

CLASS_TO_BIN = {
    "battery":    "hazardous",
    "biological": "wet",
    "cardboard":  "recycle",
    "glass":      "recycle",
    "metal":      "recycle",
    "paper":      "recycle",
    "plastic":    "recycle",
    "clothes":    "general",
    "shoes":      "general",
    "trash":      "general",
}

BIN_LABELS = {
    "general":   "General",
    "recycle":   "Recycle",
    "wet":       "Wet",
    "hazardous": "Hazardous",
}

BIN_COLORS_HEX = {
    "general":   "#0000FF",
    "recycle":   "#FFE600",
    "wet":       "#00B400",
    "hazardous": "#FF0000",
}

BIN_COLORS_BGR = {
    "general":   (255, 0, 0),
    "recycle":   (0, 230, 255),
    "wet":       (0, 180, 0),
    "hazardous": (0, 0, 255),
}

# ====================================
# App
# ====================================

app = FastAPI(
    title="Garbage Classification API",
    description="Upload an image to classify garbage and get bin recommendation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model = None


@app.on_event("startup")
def startup():
    global model
    # โหลดโมเดล
    print(f"Loading classification model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"Model loaded! Classes: {list(model.names.values())}")
    # สร้างโฟลเดอร์เก็บรูป upload
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    print(f"Upload dir ready: {UPLOAD_DIR}/")
    # สร้างตาราง database (ถ้ายังไม่มี)
    try:
        init_db()
        print("Database connected!")
    except Exception as e:
        print(f"Database not available: {e}")
        print("API will work without database logging.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# ====================================
# Resource metrics (วัดจากในคอนเทนเนอร์จริง)
# ====================================

APP_START = time.time()
predictions_served = 0
MB = 1024 * 1024


def _read_int(path):
    """อ่านค่าตัวเลขจากไฟล์ cgroup คืน None ถ้าไม่มีหรือไม่จำกัด"""
    try:
        with open(path) as f:
            v = f.read().strip().split()[0]
        return None if v in ("max", "-1") else int(v)
    except Exception:
        return None


def _container_memory():
    """คืน (ใช้ไป, ลิมิต) เป็น bytes — อ่านจาก cgroup ก่อน ถ้าไม่มีใช้ค่าทั้งเครื่อง
    สำคัญ: ในคอนเทนเนอร์ psutil จะเห็นแรมของ 'เครื่องโฮสต์' ไม่ใช่โควตาที่เราได้จริง"""
    limit = (_read_int("/sys/fs/cgroup/memory.max")
             or _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    used = (_read_int("/sys/fs/cgroup/memory.current")
            or _read_int("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
    vm = psutil.virtual_memory()
    if not limit or limit > vm.total:      # ค่ามหาศาล = ไม่ได้จำกัด
        limit = vm.total
    if used is None:
        used = vm.total - vm.available
    return used, limit


def _cpu_available():
    """จำนวน vCPU ที่คอนเทนเนอร์ใช้ได้จริง (cgroup quota) ไม่ใช่จำนวนคอร์ของโฮสต์"""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return round(int(quota) / int(period), 2)
    except Exception:
        pass
    return psutil.cpu_count()


@app.get("/metrics")
def metrics():
    """ทรัพยากรที่ใช้จริงบน server ที่ deploy อยู่ (CPU / RAM / Storage)"""
    proc = psutil.Process()
    mem_used, mem_limit = _container_memory()
    disk = psutil.disk_usage(os.path.abspath(os.sep))
    return {
        "cpu": {
            "percent": psutil.cpu_percent(interval=0.3),
            "cores_available": _cpu_available(),
        },
        "memory": {
            "process_mb": round(proc.memory_info().rss / MB, 1),
            "used_mb": round(mem_used / MB, 1),
            "limit_mb": round(mem_limit / MB, 1),
            "percent": round(mem_used / mem_limit * 100, 1),
        },
        "storage": {
            "used_mb": round(disk.used / MB, 1),
            "total_mb": round(disk.total / MB, 1),
            "percent": disk.percent,
        },
        "uptime_seconds": round(time.time() - APP_START),
        "predictions_served": predictions_served,
        "platform": {
            "system": platform.system(),
            "python": platform.python_version(),
        },
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), location: str = Query(None)):
    """Upload image, get JSON classification result"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "Could not decode image")

    global predictions_served
    predictions_served += 1

    cls_name, conf, bin_type = _get_prediction(image)

    if cls_name is None:
        return {"class": None, "confidence": 0, "bin": None, "bin_label": None, "bin_color": None}

    result = {
        "class": cls_name,
        "confidence": round(conf, 4),
        "bin": bin_type,
        "bin_label": BIN_LABELS[bin_type],
        "bin_color": BIN_COLORS_HEX[bin_type],
        "prediction_id": None,
    }

    # ตั้งชื่อไฟล์ไว้ก่อน (ยังไม่เซฟลง uploads/ — รอ user กด feedback ก่อน)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        ext = ".jpg"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_filename = f"{ts}_{cls_name}_{uuid.uuid4().hex[:8]}{ext}"

    # บันทึก metadata ลง PostgreSQL (ยังไม่เซฟรูป)
    try:
        db = next(get_db())
        prediction = save_prediction(
            db=db,
            class_name=cls_name,
            confidence=round(conf, 4),
            bin_type=bin_type,
            bin_label=BIN_LABELS[bin_type],
            image_filename=saved_filename,
            location=location,
        )
        result["prediction_id"] = prediction.id
        # พักรูปไว้ในหน่วยความจำ รอ user กด feedback ค่อยเซฟลง uploads/
        pending_images[prediction.id] = (contents, saved_filename)
        # กันหน่วยความจำบวม: เก็บรูปที่ยังรอ feedback ไม่เกิน 500
        if len(pending_images) > 500:
            pending_images.pop(next(iter(pending_images)))
    except Exception:
        pass  # ถ้า DB ไม่พร้อม ก็ข้ามไป ไม่ block การทำนาย

    return result


def _get_prediction(image):
    """Run classification inference and return (cls_name, confidence, bin_type)"""
    # Model นี้ train บนภาพต้นฉบับ (raw) จึง predict บนภาพต้นฉบับโดยตรง
    # ไม่ทำ CLAHE/GrabCut เพื่อให้ตรงกับ training distribution
    results = model.predict(image, verbose=False)
    probs = results[0].probs

    if probs is None:
        return None, 0, None

    top1_idx = int(probs.top1)
    top1_conf = float(probs.top1conf)
    cls_name = model.names[top1_idx]

    if top1_conf < CONF_THRESHOLD:
        return None, 0, None

    bin_type = CLASS_TO_BIN.get(cls_name, "general")
    return cls_name, top1_conf, bin_type


def _draw_result(image, cls_name, conf, bin_type):
    """Draw classification label on image"""
    img = image.copy()
    h, w = img.shape[:2]

    if cls_name is None:
        return img

    color = BIN_COLORS_BGR[bin_type]
    bin_label = BIN_LABELS[bin_type]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.7, min(w, h) / 500)
    thickness = max(2, int(font_scale * 2.5))
    text = f"{cls_name} {conf:.0%} | {bin_label}"
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Label bar at top
    bar_h = text_h + 24
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), color, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    text_x = (w - text_w) // 2
    text_y = (bar_h + text_h) // 2
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    return img


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Upload image, get back image with classification overlay"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "Could not decode image")

    cls_name, conf, bin_type = _get_prediction(image)
    result_img = _draw_result(image, cls_name, conf, bin_type)

    _, buffer = cv2.imencode(".jpg", result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")


# ====================================
# Feedback
# ====================================

class FeedbackRequest(BaseModel):
    prediction_id: int
    is_correct: str           # "correct" or "incorrect"
    correct_class: Optional[str] = None  # ถ้าผิด ระบุ class ที่ถูกต้อง


@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    """ส่ง feedback ว่าทำนายถูกหรือผิด"""
    if feedback.is_correct not in ("correct", "incorrect"):
        raise HTTPException(400, "is_correct must be 'correct' or 'incorrect'")

    try:
        db = next(get_db())
        prediction = update_feedback(
            db=db,
            prediction_id=feedback.prediction_id,
            is_correct=feedback.is_correct,
            correct_class=feedback.correct_class,
        )
        if prediction is None:
            raise HTTPException(404, "Prediction not found")

        # เซฟรูปลง uploads/ ตอนนี้ (หลัง user กด confirm/feedback แล้วเท่านั้น)
        pend = pending_images.pop(feedback.prediction_id, None)
        if pend is not None:
            contents, fname = pend
            try:
                with open(os.path.join(UPLOAD_DIR, fname), "wb") as f:
                    f.write(contents)
            except Exception:
                pass  # เซฟไม่ได้ก็ไม่ block feedback

        return {"status": "ok", "prediction": prediction.to_dict()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Database error: {e}")


@app.get("/classes")
def list_classes():
    """ดูรายการ class ทั้งหมด (สำหรับ feedback dropdown)"""
    return {
        "classes": [
            {"name": cls, "bin": bin_type, "bin_label": BIN_LABELS[bin_type]}
            for cls, bin_type in CLASS_TO_BIN.items()
        ]
    }


# ====================================
# Prediction History & Stats
# ====================================

@app.get("/predictions")
def list_predictions(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    """ดูประวัติการทำนาย"""
    try:
        db = next(get_db())
        predictions = get_predictions(db, limit=limit, offset=offset)
        return {"predictions": [p.to_dict() for p in predictions]}
    except Exception as e:
        raise HTTPException(500, f"Database error: {e}")


@app.get("/stats")
def prediction_stats():
    """ดูสถิติการทำนายทั้งหมด"""
    try:
        db = next(get_db())
        return get_stats(db)
    except Exception as e:
        raise HTTPException(500, f"Database error: {e}")


# ====================================
# PWA Static Files (ต้องอยู่ล่างสุด)
# ====================================
if os.path.exists("pwa"):
    app.mount("/", StaticFiles(directory="pwa", html=True), name="pwa")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
