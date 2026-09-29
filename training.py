"""
training.py
===========
Train YOLOv8 Classification Model สำหรับ Garbage Detection

วิธีใช้:
    python training.py            # Train ด้วย processed dataset (CLAHE + GrabCut)
    python training.py --raw      # Train ด้วย raw dataset (no image processing) — สำหรับ A/B test

หมายเหตุ:
    - GPU แนะนำ: NVIDIA RTX (CUDA support)
"""

import os
import sys
import torch
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path

# Import config
try:
    from config import YOLOGarbageConfig
    print("✅ Config imported")
except ImportError:
    print("❌ ไม่พบไฟล์ config.py")
    exit(1)


def check_system():
    """เช็คระบบและ GPU"""
    print("\n" + "=" * 70)
    print("  STEP 1: เช็คระบบและ GPU")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        device = 'cuda'
    else:
        print("\n  ไม่พบ GPU - จะใช้ CPU (ช้ามาก)")
        device = 'cpu'

    print(f"  PyTorch: {torch.__version__}")
    return device


def check_cls_dataset(config):
    """เช็คว่า classification dataset พร้อมใช้งาน"""
    print("\n" + "=" * 70)
    print("  STEP 2: เช็ค Classification Dataset")
    print("=" * 70)

    cls_dir = Path(config.CLS_DATA_DIR)
    train_dir = cls_dir / 'train'
    val_dir = cls_dir / 'val'

    if not train_dir.exists() or not val_dir.exists():
        print(f"\n  ไม่พบ {config.CLS_DATA_DIR}/")
        print("  กรุณารัน: python prepare_cls_dataset.py ก่อน")
        return False

    # นับ class folders
    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    val_classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])

    print(f"\n  Train classes: {len(train_classes)}")
    print(f"  Val classes:   {len(val_classes)}")

    total_train = 0
    total_val = 0
    for cls in train_classes:
        t = len(list((train_dir / cls).glob('*.[jp][pn]g')))
        v = len(list((val_dir / cls).glob('*.[jp][pn]g'))) if (val_dir / cls).exists() else 0
        print(f"    {cls:<12}: {t:>5} train / {v:>4} val")
        total_train += t
        total_val += v

    print(f"    {'รวม':<12}: {total_train:>5} train / {total_val:>4} val")
    print("\n  Dataset พร้อมใช้งาน!")
    return True


def train_cls_model(config, device):
    """เทรน classification model"""
    print("\n" + "=" * 70)
    print("  STEP 3: เทรน Classification Model")
    print("=" * 70)

    # โหลดโมเดล
    model_name = config.CLS_MODEL
    print(f"\n  โหลดโมเดล: {model_name}")
    model = YOLO(model_name)
    print(f"  Parameters: {sum(p.numel() for p in model.model.parameters()) / 1e6:.1f}M")

    # project_name จะถูกตั้งภายหลังตาม mode (raw/processed)

    print(f"\n  Epochs: {config.EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Image Size: {config.IMG_SIZE}")
    print(f"  Patience: {config.PATIENCE}")
    print(f"  Device: {device.upper()}")

    print(f"\n{'=' * 70}")
    print("  กำลังเทรน... (Ctrl+C เพื่อหยุด)")
    print("=" * 70 + "\n")

    # เลือก dataset และตั้งชื่อ run ตาม mode
    # ใช้ --raw flag เพื่อ train ด้วย raw dataset (no image processing)
    use_raw = "--raw" in sys.argv
    processed_dir = config.CLS_DATA_DIR + "_processed"

    if use_raw:
        data_dir = config.CLS_DATA_DIR
        project_name = "garbage_cls_no_processing"
        print(f"  ⚠️  RAW MODE: Training without image processing")
        print(f"  Dataset: {data_dir}")
    elif Path(processed_dir).exists():
        data_dir = processed_dir
        project_name = "garbage_cls_with_processing"
        print(f"  ✅ PROCESSED MODE: Using CLAHE + GrabCut dataset")
        print(f"  Dataset: {data_dir}")
    else:
        data_dir = config.CLS_DATA_DIR
        project_name = "garbage_cls_no_processing"
        print(f"  ⚠️  Using raw dataset: {data_dir}")

    try:
        results = model.train(
            data=data_dir,
            epochs=config.EPOCHS,
            imgsz=config.IMG_SIZE,
            batch=config.BATCH_SIZE,
            lr0=config.LEARNING_RATE,
            optimizer=config.OPTIMIZER,
            device=device,
            project=config.RUNS_DIR,
            name=project_name,
            exist_ok=True,
            pretrained=True,
            patience=config.PATIENCE,
            save=True,
            save_period=10,
            plots=True,
            verbose=True,
            # Augmentation
            degrees=config.DEGREES,
            translate=config.TRANSLATE,
            scale=config.SCALE,
            fliplr=config.FLIPLR,
            hsv_h=config.HSV_H,
            hsv_s=config.HSV_S,
            hsv_v=config.HSV_V,
            erasing=config.ERASING,
            # Improvements
            label_smoothing=config.LABEL_SMOOTHING,
            cos_lr=config.COS_LR,
        )

        print("\n" + "=" * 70)
        print("  Training เสร็จสิ้น!")
        print("=" * 70)

        # แสดงผลลัพธ์
        best_path = Path(config.RUNS_DIR) / project_name / 'weights' / 'best.pt'
        print(f"\n  Best model: {best_path}")

        # ประเมินผล
        print("\n  กำลังประเมินผล...")
        metrics = model.val()
        print(f"  Top-1 Accuracy: {metrics.top1:.4f} ({metrics.top1 * 100:.2f}%)")
        print(f"  Top-5 Accuracy: {metrics.top5:.4f} ({metrics.top5 * 100:.2f}%)")

        return model, project_name

    except KeyboardInterrupt:
        print("\n\n  Training ถูกหยุดโดยผู้ใช้")
        return None, project_name

    except Exception as e:
        print(f"\n\n  เกิดข้อผิดพลาด: {e}")
        return None, project_name


def main():
    print("=" * 70)
    print("  GARBAGE CLASSIFICATION - TRAINING")
    print("=" * 70)

    config = YOLOGarbageConfig()
    device = check_system()

    if not check_cls_dataset(config):
        return

    model, project_name = train_cls_model(config, device)

    if model is None:
        print(f"\n  Training ไม่เสร็จ ดูที่: runs/{project_name}/weights/")
        return

    print("\n" + "=" * 70)
    print("  เสร็จสมบูรณ์!")
    print(f"  ทดสอบ: python test_detect.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
