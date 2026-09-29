"""
เก็บผลการทำนายและ feedback ของผู้ใช้ลง PostgreSQL

ผลการทำนายทุกครั้งจะถูกบันทึกไว้ก่อน แล้วค่อยอัปเดตด้วยคำตอบของผู้ใช้
ว่าทำนายถูกหรือผิด ข้อมูลชุดนี้คือวัตถุดิบของการ retrain รอบถัดไป
"""

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, func
from sqlalchemy.orm import Session, declarative_base, sessionmaker

log = logging.getLogger(__name__)


def _build_database_url() -> str:
    """Railway / Render / Heroku ตั้ง DATABASE_URL ให้เองเมื่อเพิ่ม PostgreSQL
    ถ้าไม่มีก็ประกอบจากตัวแปรแยกสำหรับรันบนเครื่องตัวเอง"""
    url = os.getenv("DATABASE_URL", "").strip()

    if not url:
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        name = os.getenv("DB_NAME", "garbage_db")
        url = f"postgresql://{user}:{password}@{host}:{port}/{name}"

    # ผู้ให้บริการบางเจ้าส่งมาเป็น postgres:// ซึ่ง SQLAlchemy 2 ไม่รู้จัก
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # ระบุ driver ให้ชัด: SQLAlchemy 2.1 เปลี่ยนค่าเริ่มต้นของ postgresql://
    # จาก psycopg2 ไปเป็น psycopg (v3) ซึ่งคนละแพ็กเกจกับที่ติดตั้งไว้
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)

    return url


DATABASE_URL = _build_database_url()

# pool_pre_ping กันคอนเนกชันค้างเมื่อฐานข้อมูลคลาวด์ตัดการเชื่อมต่อที่ว่างนาน
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def _utc_now() -> datetime:
    """เวลา UTC แบบ naive ให้ตรงกับชนิดคอลัมน์ DateTime ที่ใช้อยู่"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Prediction(Base):
    """หนึ่งแถวคือการทำนายหนึ่งครั้ง พร้อมคำตอบของผู้ใช้ (ถ้ามี)"""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    class_name = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    bin_type = Column(String(20), nullable=False)
    bin_label = Column(String(20), nullable=False)
    image_filename = Column(String(255), nullable=True)
    location = Column(String(100), nullable=True)

    # ผู้ใช้ยืนยันแล้วหรือยัง: "correct" / "incorrect" / None (ยังไม่ตอบ)
    is_correct = Column(String(10), nullable=True)
    # ถ้าทำนายผิด ผู้ใช้ระบุว่าที่ถูกคือคลาสไหน
    correct_class = Column(String(50), nullable=True)

    created_at = Column(DateTime, default=_utc_now)

    def to_dict(self) -> dict:
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


def init_db() -> None:
    """สร้างตารางถ้ายังไม่มี"""
    Base.metadata.create_all(bind=engine)
    log.info("Database ready")


@contextmanager
def session_scope() -> Iterator[Session]:
    """เปิด session และปิดให้เสมอ แม้เกิด error ระหว่างทาง"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction(
    db: Session,
    class_name: str,
    confidence: float,
    bin_type: str,
    bin_label: str,
    image_filename: Optional[str] = None,
    location: Optional[str] = None,
) -> Prediction:
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


def update_feedback(
    db: Session,
    prediction_id: int,
    is_correct: str,
    correct_class: Optional[str] = None,
) -> Optional[Prediction]:
    """บันทึกคำตอบของผู้ใช้ คืน None ถ้าไม่พบ prediction นั้น"""
    prediction = db.get(Prediction, prediction_id)
    if prediction is None:
        return None

    prediction.is_correct = is_correct
    prediction.correct_class = correct_class
    db.commit()
    db.refresh(prediction)
    return prediction


def get_predictions(db: Session, limit: int = 50, offset: int = 0) -> list[Prediction]:
    return (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_stats(db: Session) -> dict:
    """สรุปภาพรวม: ทำนายไปกี่ครั้ง แม่นแค่ไหนตาม feedback และกระจายตัวของแต่ละคลาส"""
    total = db.query(func.count(Prediction.id)).scalar() or 0

    by_class = (
        db.query(
            Prediction.class_name,
            func.count(Prediction.id).label("count"),
            func.avg(Prediction.confidence).label("avg_confidence"),
        )
        .group_by(Prediction.class_name)
        .order_by(func.count(Prediction.id).desc())
        .all()
    )

    by_bin = (
        db.query(Prediction.bin_type, func.count(Prediction.id).label("count"))
        .group_by(Prediction.bin_type)
        .order_by(func.count(Prediction.id).desc())
        .all()
    )

    def _count_feedback(value: str) -> int:
        return db.query(func.count(Prediction.id)).filter(
            Prediction.is_correct == value
        ).scalar() or 0

    correct = _count_feedback("correct")
    incorrect = _count_feedback("incorrect")
    answered = correct + incorrect

    return {
        "total_predictions": total,
        "feedback": {
            "correct": correct,
            "incorrect": incorrect,
            "no_feedback": total - answered,
            # คิดจากเฉพาะครั้งที่ผู้ใช้ตอบ ไม่ใช่จากการทำนายทั้งหมด
            "accuracy": round(correct / answered * 100, 1) if answered else 0.0,
        },
        "by_class": [
            {
                "class_name": row.class_name,
                "count": row.count,
                "avg_confidence": round(float(row.avg_confidence), 4),
            }
            for row in by_class
        ],
        "by_bin": [{"bin_type": row.bin_type, "count": row.count} for row in by_bin],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
