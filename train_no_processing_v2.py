"""
train_no_processing_v2.py
เทรนโมเดล no_processing ใหม่บน dataset ปัจจุบัน (ที่มี data เพิ่ม)
เซฟเป็นชื่อใหม่ garbage_cls_no_processing_v2  -> ไม่ทับของเก่า

โมเดลเดิมที่เก็บไว้:
  - garbage_cls_no_processing     (96.14% - ที่ deploy อยู่)
  - garbage_cls_no_processing_v1  (เก่ากว่า)

รัน:
    python train_no_processing_v2.py
โมเดลใหม่: runs/classify/runs/garbage_cls_no_processing_v2/weights/best.pt
"""
import os, sys, io
import torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA = "garbage_dataset_cls"                 # raw dataset ปัจจุบัน
NAME = "garbage_cls_no_processing_v2"        # ชื่อใหม่ ไม่ทับของเก่า


def main():
    from config import YOLOGarbageConfig
    from ultralytics import YOLO

    cfg = YOLOGarbageConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # กันทับของเก่าโดยไม่ตั้งใจ
    out_dir = os.path.join(cfg.RUNS_DIR, "classify", "runs", NAME)
    if os.path.exists(os.path.join(out_dir, "weights", "best.pt")):
        print(f"⚠️  มี {NAME} อยู่แล้ว — จะเขียนทับ (exist_ok=True)")

    if not os.path.isdir(os.path.join(DATA, "train")):
        print(f"❌ ไม่พบ {DATA}/train"); return

    print("=" * 62)
    print(f"  Train no_processing v2  |  device={device.upper()}")
    print(f"  data={DATA}  epochs={cfg.EPOCHS}  imgsz={cfg.IMG_SIZE}  opt={cfg.OPTIMIZER}")
    print(f"  บันทึกเป็น: {NAME}  (ของเก่าไม่ถูกแตะ)")
    print("=" * 62)

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
    print(f"\n  โมเดลใหม่: runs/classify/runs/{NAME}/weights/best.pt")
    print(f"  วัด F1 + confusion matrix ต่อ:")
    print(f"    python eval_f1.py runs/classify/runs/{NAME}/weights/best.pt {DATA}")


if __name__ == "__main__":
    main()
