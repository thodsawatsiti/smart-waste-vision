"""
train_material_patches.py
เทรนด้วย recipe เดียวกับโมเดลอื่นทุกตัว (150 epoch, yolo11n-cls, AdamW) แต่ข้อมูล train
เป็น patch ที่ครอปซูมเข้าไปที่พื้นผิว (garbage_dataset_cls_patches) ส่วน val ยังเป็นรูปเต็ม
เพื่อเทียบผลกับ no_processing / with_processing แบบ apples-to-apples

ต้องรัน preprocess_material_patches.py ให้เสร็จก่อน
รัน:
    python train_material_patches.py
โมเดลจะถูกบันทึกที่: runs/classify/runs/garbage_cls_material_patches/weights/best.pt
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")

DATA = "garbage_dataset_cls_patches"
NAME = "garbage_cls_material_patches"


def main():
    from config import YOLOGarbageConfig
    from ultralytics import YOLO

    cfg = YOLOGarbageConfig()

    if not os.path.isdir(os.path.join(DATA, "train")):
        print(f"ไม่พบ {DATA}/ — รัน preprocess_material_patches.py ก่อน")
        return

    model = YOLO(cfg.CLS_MODEL)
    model.train(
        data=DATA,
        epochs=cfg.EPOCHS,
        imgsz=cfg.IMG_SIZE,
        batch=8,  # กันปัญหา CUDA out of memory (เจอมาแล้วรอบก่อน)
        lr0=cfg.LEARNING_RATE,
        optimizer=cfg.OPTIMIZER,
        device=0,
        project="runs/classify/runs",
        name=NAME,
        exist_ok=True,
        pretrained=True,
        patience=cfg.PATIENCE,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        degrees=cfg.DEGREES,
        translate=cfg.TRANSLATE,
        scale=cfg.SCALE,
        fliplr=cfg.FLIPLR,
        hsv_h=cfg.HSV_H,
        hsv_s=cfg.HSV_S,
        hsv_v=cfg.HSV_V,
        erasing=cfg.ERASING,
        label_smoothing=cfg.LABEL_SMOOTHING,
        cos_lr=cfg.COS_LR,
    )

    print("\n  ประเมินผล...")
    m = model.val()
    print(f"  Top-1: {m.top1*100:.2f}%   Top-5: {m.top5*100:.2f}%")
    print(f"\n  โมเดล: runs/classify/runs/{NAME}/weights/best.pt")


if __name__ == "__main__":
    main()
