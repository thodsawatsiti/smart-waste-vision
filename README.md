# Smart Waste Vision

ระบบจำแนกประเภทขยะด้วยปัญญาประดิษฐ์ผ่านกล้องมือถือ ช่วยให้ผู้ใช้คัดแยกขยะได้ถูกต้องตั้งแต่ต้นทาง
โดยถ่ายรูปขยะแล้วระบบจะบอกว่าควรทิ้งถังสีอะไร

> A YOLOv11-based Progressive Web Application for point-of-disposal waste classification.

---

## ภาพรวม

| | |
|---|---|
| โมเดล | YOLOv11n-cls (classification) |
| ประเภทขยะ | 10 คลาส → 4 ถังปลายทาง |
| ความแม่นยำ | 96.1% บนชุดตรวจสอบ |
| ส่วนหน้า | Progressive Web App (เปิดผ่านเบราว์เซอร์ ไม่ต้องติดตั้ง) |
| ส่วนหลัง | FastAPI + PostgreSQL |

### การจับคู่ประเภทขยะกับถัง

| คลาส | ถังปลายทาง |
|---|---|
| battery | ขยะอันตราย (ถังสีแดง) |
| biological | ขยะเปียก (ถังสีเขียว) |
| cardboard, glass, metal, paper, plastic | ขยะรีไซเคิล (ถังสีเหลือง) |
| clothes, shoes, trash | ขยะทั่วไป (ถังสีน้ำเงิน) |

---

## จุดเด่น: เรียนรู้ต่อเนื่องจากผู้ใช้ (Feedback Loop)

ผู้ใช้กดยืนยันได้ว่าผลการทำนายถูกหรือผิด ระบบจะเก็บภาพที่ยืนยันแล้วลงฐานข้อมูล
เพื่อนำกรณีที่ทำนายผิดไปฝึกโมเดลซ้ำ

ผลจากการฝึกซ้ำรอบแรก — ทดสอบกับภาพถ่ายมือถือจริงที่โมเดลไม่เคยเห็น:

| | ก่อน Retrain | หลัง Retrain |
|---|---|---|
| ทายถูก | 6 / 16 | **16 / 16** |
| ความแม่นยำ | 37.5% | **100%** |

---

## การติดตั้งและใช้งาน

### รันบนเครื่องตัวเอง

```bash
pip install -r requirements.txt
cp .env.example .env        # แล้วแก้ค่าในไฟล์ .env
uvicorn api:app --reload
```

เปิด http://localhost:8000

### รันด้วย Docker

```bash
docker build -t smart-waste-vision .
docker run -p 8000:8000 --env-file .env smart-waste-vision
```

### Deploy บน Railway

1. Push repo นี้ขึ้น GitHub
2. สร้างโปรเจกต์ใหม่บน Railway แล้วเลือก **Deploy from GitHub repo**
3. เพิ่ม **PostgreSQL** ในโปรเจกต์ — Railway จะตั้งตัวแปร `DATABASE_URL` ให้อัตโนมัติ
4. Railway อ่าน `Dockerfile` และกำหนด `PORT` ให้เอง ไม่ต้องตั้งค่าเพิ่ม

> ต้องใช้ HTTPS เบราว์เซอร์ถึงจะอนุญาตให้เปิดกล้อง — Railway ให้โดเมน HTTPS มาอยู่แล้ว

---

## API

| Endpoint | หน้าที่ |
|---|---|
| `POST /predict` | ส่งภาพ คืนประเภทขยะและถังปลายทาง |
| `POST /feedback` | บันทึกการยืนยันผลจากผู้ใช้ |
| `GET /metrics` | ทรัพยากรที่ใช้บนเซิร์ฟเวอร์ (CPU / RAM / Storage) |
| `GET /stats` | สถิติการทำนายสะสม |
| `GET /health` | ตรวจสถานะระบบ |
| `GET /docs` | เอกสาร API (Swagger UI) |

---

## ตัวแปรสภาพแวดล้อม

| ตัวแปร | คำอธิบาย |
|---|---|
| `DATABASE_URL` | URL ของ PostgreSQL (Railway ตั้งให้อัตโนมัติ) |
| `MODEL_PATH` | ตำแหน่งไฟล์โมเดล ค่าเริ่มต้น `models/best.pt` |
| `PORT` | พอร์ตที่ให้บริการ ค่าเริ่มต้น 8000 |

ดูตัวอย่างทั้งหมดที่ [`.env.example`](.env.example)

> ⚠️ อย่า commit ไฟล์ `.env` หรือใส่รหัสผ่านลงในโค้ด

---

## โครงสร้างโปรเจกต์

```
api.py                  FastAPI server + endpoints
database.py             โมเดลข้อมูลและการเชื่อมต่อ PostgreSQL
image_processing.py     ขั้นตอนประมวลผลภาพ (CLAHE + GrabCut) สำหรับการทดลอง
pwa/                    ส่วนหน้า Progressive Web App
models/best.pt          โมเดลที่ใช้งานจริง
train_*.py              สคริปต์ฝึกโมเดลในแต่ละการทดลอง
preprocess_*.py         สคริปต์เตรียมชุดข้อมูล
eval_*.py, make_*.py    สคริปต์ประเมินผลและสร้างกราฟ
```

ชุดข้อมูลและผลการเทรน (`garbage_dataset*/`, `runs/`, `uploads/`) ไม่ได้รวมไว้ใน repo
เนื่องจากมีขนาดใหญ่และมีข้อมูลส่วนบุคคล

---


