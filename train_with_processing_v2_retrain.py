"""
train_with_processing_v2_retrain.py
เทรน CLAHE+GrabCut (ไม่มี fallback) ใหม่อีกรอบ ด้วย batch ใหญ่ขึ้น + epoch มากขึ้น
(รอบก่อน batch=8 เพราะตอนนั้น VRAM ถูกโปรแกรมพื้นหลังแย่งไปเยอะ ตอนนี้ว่างแล้วเลยเพิ่มได้)
ข้อมูล: garbage_dataset_cls_processed_v2 (เดิม ไม่ได้ preprocess ใหม่)
บันทึกคนละโฟลเดอร์กับรอบก่อน (garbage_cls_with_processing_v2) เพื่อเทียบผลได้
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from ultralytics import YOLO
from config import YOLOGarbageConfig

config = YOLOGarbageConfig()
DATA = "garbage_dataset_cls_processed_v2"
RUN_NAME = "garbage_cls_with_processing_v2_retrain"

def main():
    model = YOLO(config.CLS_MODEL)  # yolo11n-cls.pt, pretrained
    model.train(
        data=DATA,
        epochs=250,       # เดิม 150 (เพิ่มเพดาน)
        patience=40,       # เดิม 30 (ต้องเพิ่มด้วย ไม่งั้น early stop จุดเดิม ไม่ทันได้ใช้ epoch เพิ่ม)
        imgsz=config.IMG_SIZE,
        batch=24,          # เดิม 8 (ตอนนี้ VRAM ว่าง ~6.2GB เพิ่มได้)
        lr0=config.LEARNING_RATE,
        optimizer=config.OPTIMIZER,
        device=0,
        project="runs/classify/runs",
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
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
