"""
train_clahe_sam.py
เทรนโมเดลบน dataset ที่ผ่าน CLAHE + SAM (garbage_dataset_cls_clahe_sam)
ใช้ config เดียวกับโมเดลอื่น (640px, 150ep, AdamW, augmentation ครบ) -> เทียบได้ตรง

ต้องรัน preprocess_clahe_sam.py ให้เสร็จก่อน
รัน:
    python train_clahe_sam.py
โมเดลจะถูกบันทึกที่: runs/classify/runs/garbage_cls_clahe_sam/weights/best.pt
"""
import os, sys, io
import torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA = "garbage_dataset_cls_clahe_sam"
NAME = "garbage_cls_clahe_sam"


def main():
    from config import YOLOGarbageConfig
    from ultralytics import YOLO

    cfg = YOLOGarbageConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isdir(os.path.join(DATA, "train")):
        print(f"❌ ไม่พบ {DATA}/ — รัน preprocess_clahe_sam.py ก่อน")
        return

    print("=" * 60)
    print(f"  Train CLAHE+SAM model  |  device={device.upper()}")
    print(f"  data={DATA}  epochs={cfg.EPOCHS}  imgsz={cfg.IMG_SIZE}  opt={cfg.OPTIMIZER}")
    print("=" * 60)

    model = YOLO(cfg.CLS_MODEL)
    model.train(
        data=DATA,
        epochs=cfg.EPOCHS,
        imgsz=cfg.IMG_SIZE,
        batch=cfg.BATCH_SIZE,
        lr0=cfg.LEARNING_RATE,
        optimizer=cfg.OPTIMIZER,
        device=device,
        project=cfg.RUNS_DIR,
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
    print(f"  วัด F1 ต่อ: python eval_f1.py runs/classify/runs/{NAME}/weights/best.pt {DATA}")


if __name__ == "__main__":
    main()
