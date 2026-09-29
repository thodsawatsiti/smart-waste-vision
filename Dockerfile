FROM python:3.11-slim

WORKDIR /app

# ไลบรารีระบบที่ OpenCV ต้องใช้
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/tmp/hf \
    PYTHONUNBUFFERED=1

# ติดตั้ง torch แบบ CPU-only ก่อน (เล็กกว่าแบบ CUDA หลาย GB และเซิร์ฟเวอร์ไม่มี GPU)
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py database.py ./
COPY models/ ./models/
COPY pwa/ ./pwa/

# Railway กำหนดพอร์ตมาทาง $PORT
ENV PORT=8000
EXPOSE 8000
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT}
