"""
database.py
===========
PostgreSQL Database สำหรับเก็บ Prediction Logs

ใช้ SQLAlchemy ORM + psycopg2
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base

# ====================================
# Config - อ่านจาก environment variable เท่านั้น (ไม่ฝังรหัสผ่านในโค้ด)
# ====================================

# Railway / Render / Heroku จะตั้ง DATABASE_URL ให้อัตโนมัติเมื่อเพิ่ม PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

if not DATABASE_URL:
    # กรณีรันบนเครื่องตัวเอง ประกอบ URL จากตัวแปรแยก
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")      # ต้องตั้งเอง ไม่มีค่า default
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "garbage_db")
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# บางผู้ให้บริการส่งมาเป็น postgres:// ซึ่ง SQLAlchemy 2 ไม่รองรับ
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# ====================================
# SQLAlchemy Setup
# ====================================

# pool_pre_ping: กันคอนเนกชันหลุดเมื่อฐานข้อมูลคลาวด์ตัดการเชื่อมต่อที่ว่างนาน
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# ====================================
# Models (ตาราง)
# ====================================

class Prediction(Base):
    """ตารางเก็บผลการทำนายทุกครั้ง"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    class_name = Column(String(50), nullable=False)       # เช่น "plastic"
    confidence = Column(Float, nullable=False)             # เช่น 0.9639
    bin_type = Column(String(20), nullable=False)          # เช่น "recycle"
    bin_label = Column(String(20), nullable=False)         # เช่น "Recycle"
    image_filename = Column(String(255), nullable=True)    # ชื่อไฟล์รูป
    location = Column(String(100), nullable=True)          # จุดทิ้งขยะ (ถ้ามี)
    is_correct = Column(String(10), nullable=True)         # "correct" / "incorrect" / None
    correct_class = Column(String(50), nullable=True)      # class ที่ถูกต้อง (ถ้าทำนายผิด)
    created_at = Column(DateTime, default=datetime.utcnow) # เวลาที่ทำนาย

    def to_dict(self):
        return {
            "id": self.id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bin_type": self.bin_type,
            "bin_label": self.bin_label,
            "image_filename": self.image_filename,
            "location": self.location,
            "is_correct": self.is_correct,
            "correct_class": self.correct_class,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ====================================
# สร้างตาราง
# ====================================

def init_db():
    """สร้างตารางทั้งหมดใน database"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created!")


def get_db():
    """สร้าง database session (ใช้กับ FastAPI Depends)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ====================================
# Helper Functions
# ====================================

def save_prediction(db, class_name: str, confidence: float, bin_type: str,
                    bin_label: str, image_filename: str = None, location: str = None):
    """บันทึกผลทำนายลง database"""
    prediction = Prediction(
        class_name=class_name,
        confidence=confidence,
        bin_type=bin_type,
        bin_label=bin_label,
        image_filename=image_filename,
        location=location,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


def update_feedback(db, prediction_id: int, is_correct: str, correct_class: str = None):
    """อัพเดท feedback ว่าทำนายถูก/ผิด"""
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if prediction:
        prediction.is_correct = is_correct
        prediction.correct_class = correct_class
        db.commit()
        db.refresh(prediction)
    return prediction


def get_predictions(db, limit: int = 50, offset: int = 0):
    """ดึงประวัติการทำนาย"""
    return db.query(Prediction)\
        .order_by(Prediction.created_at.desc())\
        .offset(offset)\
        .limit(limit)\
        .all()


def get_stats(db):
    """ดึงสถิติการทำนายทั้งหมด"""
    from sqlalchemy import func

    total = db.query(func.count(Prediction.id)).scalar()

    # นับจำนวนแต่ละ class
    class_stats = db.query(
        Prediction.class_name,
        func.count(Prediction.id).label("count"),
        func.avg(Prediction.confidence).label("avg_confidence"),
    ).group_by(Prediction.class_name)\
     .order_by(func.count(Prediction.id).desc())\
     .all()

    # นับจำนวนแต่ละ bin
    bin_stats = db.query(
        Prediction.bin_type,
        func.count(Prediction.id).label("count"),
    ).group_by(Prediction.bin_type)\
     .order_by(func.count(Prediction.id).desc())\
     .all()

    # นับ feedback
    correct_count = db.query(func.count(Prediction.id))\
        .filter(Prediction.is_correct == "correct").scalar()
    incorrect_count = db.query(func.count(Prediction.id))\
        .filter(Prediction.is_correct == "incorrect").scalar()

    return {
        "total_predictions": total,
        "feedback": {
            "correct": correct_count,
            "incorrect": incorrect_count,
            "no_feedback": total - correct_count - incorrect_count,
            "accuracy": round(correct_count / max(correct_count + incorrect_count, 1) * 100, 1),
        },
        "by_class": [
            {
                "class_name": row.class_name,
                "count": row.count,
                "avg_confidence": round(float(row.avg_confidence), 4),
            }
            for row in class_stats
        ],
        "by_bin": [
            {
                "bin_type": row.bin_type,
                "count": row.count,
            }
            for row in bin_stats
        ],
    }


if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Done!")
