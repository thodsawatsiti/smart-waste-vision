"""
train_with_processing_v2.py
เทรน CLAHE+GrabCut (ไม่มี fallback) ด้วย recipe เดียวกับ config.py เป๊ะๆ
เพื่อเทียบกับ no_processing แบบ apples-to-apples
ข้อมูล: garbage_dataset_cls_processed_v2 (รวม hard examples จาก feedback loop แล้ว)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from ultralytics import YOLO
from config import YOLOGarbageConfig

config = YOLOGarbageConfig()
DATA = "garbage_dataset_cls_processed_v2"
RUN_NAME = "garbage_cls_with_processing_v2"

def main():
    model = YOLO(config.CLS_MODEL)  # yolo11n-cls.pt, pretrained
    model.train(
        data=DATA,
        epochs=config.EPOCHS,
        imgsz=config.IMG_SIZE,
        batch=8,  # ลดจาก config.BATCH_SIZE (16) เพราะ CUDA OOM ตอน validation
                  # (เครื่องมี VRAM 8GB และมีโปรแกรมพื้นหลังแย่งอยู่เยอะ)
        lr0=config.LEARNING_RATE,
        optimizer=config.OPTIMIZER,
        device=0,
        project="runs/classify/runs",
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        patience=config.PATIENCE,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        degrees=config.DEGREES,
        translate=config.TRANSLATE,
        scale=config.SCALE,
        fliplr=config.FLIPLR,
        hsv_h=config.HSV_H,
        hsv_s=config.HSV_S,
        hsv_v=config.HSV_V,
        erasing=config.ERASING,
        label_smoothing=config.LABEL_SMOOTHING,
        cos_lr=config.COS_LR,
    )
    print(f"\nเสร็จแล้ว! runs/classify/runs/{RUN_NAME}/weights/best.pt")

if __name__ == "__main__":
    main()
