"""
Smart Waste Vision — FastAPI server

รับภาพขยะจากกล้องมือถือ จำแนกประเภทด้วย YOLOv11n แล้วบอกถังปลายทาง
พร้อมเก็บคำยืนยันของผู้ใช้ไว้ใช้ retrain รอบถัดไป

รัน:  uvicorn api:app --reload
"""

import io
import logging
import os
import platform
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import psutil
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

from database import (
    get_predictions,
    get_stats,
    init_db,
    save_prediction,
    session_scope,
    update_feedback,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- config

# โมเดลที่ใช้งานจริง: เทรนบนภาพต้นฉบับ แล้ว fine-tune ด้วย hard examples จาก feedback loop
# val 95.8% | ภาพถ่ายมือถือจริงที่ไม่เคยเห็นตอนเทรน 37.5% -> 100%
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")

# ต่ำกว่านี้ถือว่าไม่มั่นใจพอ ตอบว่าไม่รู้จักดีกว่าเดาสุ่ม
CONF_THRESHOLD = 0.25

UPLOAD_DIR = "uploads"
ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".webp")

# จำนวนภาพที่พักรอ feedback ได้พร้อมกัน กันหน่วยความจำบวมถ้าไม่มีใครกดยืนยัน
MAX_PENDING_IMAGES = 500

CLASS_TO_BIN = {
    "battery": "hazardous",
    "biological": "wet",
    "cardboard": "recycle",
    "glass": "recycle",
    "metal": "recycle",
    "paper": "recycle",
    "plastic": "recycle",
    "clothes": "general",
    "shoes": "general",
    "trash": "general",
}

BIN_LABELS = {
    "general": "General",
    "recycle": "Recycle",
    "wet": "Wet",
    "hazardous": "Hazardous",
}

BIN_COLORS_HEX = {
    "general": "#0000FF",
    "recycle": "#FFE600",
    "wet": "#00B400",
    "hazardous": "#FF0000",
}

BIN_COLORS_BGR = {
    "general": (255, 0, 0),
    "recycle": (0, 230, 255),
    "wet": (0, 180, 0),
    "hazardous": (0, 0, 255),
}

# ---------------------------------------------------------------- state

model: Optional[YOLO] = None

# ภาพที่ทำนายแล้วแต่ยังรอผู้ใช้ยืนยัน {prediction_id: (bytes, filename)}
# เก็บลงดิสก์ต่อเมื่อยืนยันแล้ว จะได้ไม่สะสมภาพที่ไม่มีป้ายกำกับ
pending_images: dict[int, tuple[bytes, str]] = {}

app_started_at = time.time()
predictions_served = 0


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model

    log.info("Loading model: %s", MODEL_PATH)
    model = YOLO(MODEL_PATH)
    log.info("Model ready, classes: %s", list(model.names.values()))

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # ไม่มีฐานข้อมูลก็ยังทำนายได้ แค่เก็บ feedback ไม่ได้
    try:
        init_db()
    except Exception as exc:
        log.warning("Database unavailable, feedback will not be stored: %s", exc)

    yield


app = FastAPI(
    title="Smart Waste Vision API",
    description="จำแนกประเภทขยะจากภาพถ่าย และแนะนำถังปลายทาง",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- helpers

def _decode_upload(contents: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image")
    return image


def _classify(image: np.ndarray) -> tuple[Optional[str], float, Optional[str]]:
    """คืน (คลาส, ความเชื่อมั่น, ถัง) หรือ (None, 0, None) ถ้าไม่มั่นใจพอ

    โมเดลเทรนบนภาพต้นฉบับ จึงส่งภาพเข้าไปตรงๆ ไม่ผ่าน CLAHE/GrabCut
    เพื่อให้ตรงกับการกระจายของข้อมูลตอนเทรน
    """
    probs = model.predict(image, verbose=False)[0].probs
    if probs is None:
        return None, 0.0, None

    confidence = float(probs.top1conf)
    if confidence < CONF_THRESHOLD:
        return None, 0.0, None

    class_name = model.names[int(probs.top1)]
    return class_name, confidence, CLASS_TO_BIN.get(class_name, "general")


def _build_filename(class_name: str, original_name: Optional[str]) -> str:
    ext = os.path.splitext(original_name or "")[1].lower()
    if ext not in ALLOWED_EXT:
        ext = ".jpg"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{class_name}_{uuid.uuid4().hex[:8]}{ext}"


def _draw_result(
    image: np.ndarray, class_name: Optional[str], confidence: float, bin_type: Optional[str]
) -> np.ndarray:
    """วางแถบผลการทำนายไว้ด้านบนของภาพ"""
    if class_name is None:
        return image

    img = image.copy()
    height, width = img.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.7, min(width, height) / 500)
    thickness = max(2, int(font_scale * 2.5))
    text = f"{class_name} {confidence:.0%} | {BIN_LABELS[bin_type]}"
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    bar_height = text_height + 24
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_height), BIN_COLORS_BGR[bin_type], -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    cv2.putText(
        img,
        text,
        ((width - text_width) // 2, (bar_height + text_height) // 2),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )
    return img


# ---------------------------------------------------------------- prediction

@app.post("/predict")
async def predict(file: UploadFile = File(...), location: Optional[str] = Query(None)):
    """รับภาพ คืนประเภทขยะและถังปลายทาง"""
    global predictions_served

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    image = _decode_upload(contents)
    predictions_served += 1

    class_name, confidence, bin_type = _classify(image)
    if class_name is None:
        return {
            "class": None,
            "confidence": 0,
            "bin": None,
            "bin_label": None,
            "bin_color": None,
        }

    result = {
        "class": class_name,
        "confidence": round(confidence, 4),
        "bin": bin_type,
        "bin_label": BIN_LABELS[bin_type],
        "bin_color": BIN_COLORS_HEX[bin_type],
        "prediction_id": None,
    }

    # บันทึกเฉพาะข้อมูลลงฐานข้อมูลก่อน เพื่อให้ได้ id ไว้ผูกกับ feedback
    # ตัวภาพพักไว้ในหน่วยความจำ รอจนกว่าผู้ใช้จะยืนยัน
    filename = _build_filename(class_name, file.filename)
    try:
        with session_scope() as db:
            prediction = save_prediction(
                db=db,
                class_name=class_name,
                confidence=result["confidence"],
                bin_type=bin_type,
                bin_label=BIN_LABELS[bin_type],
                image_filename=filename,
                location=location,
            )
            result["prediction_id"] = prediction.id

        if len(pending_images) >= MAX_PENDING_IMAGES:
            pending_images.pop(next(iter(pending_images)))
        pending_images[result["prediction_id"]] = (contents, filename)
    except Exception as exc:
        # ฐานข้อมูลล่มไม่ควรทำให้ทำนายไม่ได้ แค่เก็บ feedback ไม่ได้เท่านั้น
        log.warning("Could not store prediction: %s", exc)

    return result


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """เหมือน /predict แต่คืนเป็นภาพที่วางผลการทำนายทับไว้"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    image = _decode_upload(await file.read())
    annotated = _draw_result(image, *_classify(image))

    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")


# ---------------------------------------------------------------- feedback

class FeedbackRequest(BaseModel):
    prediction_id: int
    is_correct: str  # "correct" หรือ "incorrect"
    correct_class: Optional[str] = None  # ระบุเมื่อทำนายผิด


@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    """บันทึกคำยืนยันของผู้ใช้ และเก็บภาพลงดิสก์เมื่อยืนยันแล้ว"""
    if feedback.is_correct not in ("correct", "incorrect"):
        raise HTTPException(400, "is_correct must be 'correct' or 'incorrect'")

    try:
        with session_scope() as db:
            prediction = update_feedback(
                db=db,
                prediction_id=feedback.prediction_id,
                is_correct=feedback.is_correct,
                correct_class=feedback.correct_class,
            )
            if prediction is None:
                raise HTTPException(404, "Prediction not found")
            payload = prediction.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Could not save feedback")
        raise HTTPException(500, f"Database error: {exc}") from exc

    _store_confirmed_image(feedback.prediction_id)
    return {"status": "ok", "prediction": payload}


def _store_confirmed_image(prediction_id: int) -> None:
    """เก็บภาพลงดิสก์หลังผู้ใช้ยืนยันแล้ว — ภาพชุดนี้คือข้อมูลสำหรับ retrain

    หมายเหตุ: บน hosting ที่ไม่มีดิสก์ถาวร โฟลเดอร์นี้จะหายเมื่อ restart
    ถ้าต้องการเก็บถาวรควรต่อ object storage
    """
    pending = pending_images.pop(prediction_id, None)
    if pending is None:
        return

    contents, filename = pending
    try:
        with open(os.path.join(UPLOAD_DIR, filename), "wb") as f:
            f.write(contents)
    except OSError as exc:
        # เก็บภาพไม่ได้ก็ไม่ควรทำให้ feedback ที่บันทึกไปแล้วล้มเหลว
        log.warning("Could not write %s: %s", filename, exc)


@app.get("/classes")
def list_classes():
    """รายการคลาสทั้งหมด ใช้เติม dropdown ตอนผู้ใช้แจ้งว่าทำนายผิด"""
    return {
        "classes": [
            {"name": name, "bin": bin_type, "bin_label": BIN_LABELS[bin_type]}
            for name, bin_type in CLASS_TO_BIN.items()
        ]
    }


# ---------------------------------------------------------------- history

@app.get("/predictions")
def list_predictions(
    limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)
):
    try:
        with session_scope() as db:
            return {"predictions": [p.to_dict() for p in get_predictions(db, limit, offset)]}
    except Exception as exc:
        raise HTTPException(500, f"Database error: {exc}") from exc


@app.get("/stats")
def prediction_stats():
    try:
        with session_scope() as db:
            return get_stats(db)
    except Exception as exc:
        raise HTTPException(500, f"Database error: {exc}") from exc


# ---------------------------------------------------------------- monitoring

BYTES_PER_MB = 1024 * 1024


def _read_cgroup_int(path: str) -> Optional[int]:
    """อ่านตัวเลขจากไฟล์ cgroup คืน None ถ้าไม่มีไฟล์หรือไม่ได้จำกัดไว้"""
    try:
        with open(path) as f:
            value = f.read().strip().split()[0]
    except OSError:
        return None
    return None if value in ("max", "-1") else int(value)


def _container_memory() -> tuple[int, int]:
    """คืน (ใช้ไป, ลิมิต) เป็นไบต์

    ในคอนเทนเนอร์ psutil รายงานแรมของ "เครื่องโฮสต์" ไม่ใช่โควตาที่เราได้จริง
    จึงต้องอ่านจาก cgroup ก่อน แล้วค่อย fallback
    """
    limit = _read_cgroup_int("/sys/fs/cgroup/memory.max") or _read_cgroup_int(
        "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    )
    used = _read_cgroup_int("/sys/fs/cgroup/memory.current") or _read_cgroup_int(
        "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    )

    machine = psutil.virtual_memory()
    if not limit or limit > machine.total:
        limit = machine.total
    if used is None:
        used = machine.total - machine.available

    return used, limit


def _cpu_available() -> float:
    """จำนวน vCPU ที่คอนเทนเนอร์ใช้ได้จริง ไม่ใช่จำนวนคอร์ทั้งหมดของโฮสต์"""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return round(int(quota) / int(period), 2)
    except (OSError, ValueError):
        pass
    return psutil.cpu_count()


@app.get("/metrics")
def metrics():
    """ทรัพยากรที่ใช้จริงบนเซิร์ฟเวอร์ที่ deploy อยู่"""
    memory_used, memory_limit = _container_memory()
    disk = psutil.disk_usage(os.path.abspath(os.sep))
    to_mb = lambda value: round(value / BYTES_PER_MB, 1)  # noqa: E731

    return {
        "cpu": {
            "percent": psutil.cpu_percent(interval=0.3),
            "cores_available": _cpu_available(),
        },
        "memory": {
            "process_mb": to_mb(psutil.Process().memory_info().rss),
            "used_mb": to_mb(memory_used),
            "limit_mb": to_mb(memory_limit),
            "percent": round(memory_used / memory_limit * 100, 1),
        },
        "storage": {
            "used_mb": to_mb(disk.used),
            "total_mb": to_mb(disk.total),
            "percent": disk.percent,
        },
        "uptime_seconds": round(time.time() - app_started_at),
        "predictions_served": predictions_served,
        "platform": {"system": platform.system(), "python": platform.python_version()},
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# หน้าเว็บ PWA ต้อง mount ท้ายสุด ไม่งั้นจะไปทับ path ของ API ด้านบน
if os.path.exists("pwa"):
    app.mount("/", StaticFiles(directory="pwa", html=True), name="pwa")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
