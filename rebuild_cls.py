"""
rebuild_cls.py
ล้าง garbage_dataset_cls แล้ว copy จาก source ใหม่ แบบ 80/20 train/val
"""

import os
import shutil
import random
import sys
sys.stdout.reconfigure(encoding='utf-8')

SOURCE = r"C:\Users\popjr\Desktop\garbage\garbage-detection-yolov8\garbage-classification-v2\garbage-dataset"
TARGET = r"C:\Users\popjr\Desktop\garbage\garbage-detection-yolov8\garbage_dataset_cls"
TRAIN_RATIO = 0.8
MAX_PER_CLASS = 1000
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

def get_images(folder):
    return [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in EXTENSIONS]

def main():
    classes = sorted([d for d in os.listdir(SOURCE) if os.path.isdir(os.path.join(SOURCE, d))])

    print(f"{'Class':<15} {'Total':>7} {'Train':>7} {'Val':>7}")
    print("-" * 40)

    for cls in classes:
        src_folder = os.path.join(SOURCE, cls)
        train_folder = os.path.join(TARGET, "train", cls)
        val_folder = os.path.join(TARGET, "val", cls)

        # ล้าง folder เก่า แล้วสร้างใหม่
        shutil.rmtree(train_folder, ignore_errors=True)
        shutil.rmtree(val_folder, ignore_errors=True)
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        # เอารูปทั้งหมดแล้ว shuffle แล้วตัดเหลือ MAX_PER_CLASS
        images = get_images(src_folder)
        random.shuffle(images)
        images = images[:MAX_PER_CLASS]

        # Split 80/20
        split = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:split]
        val_imgs = images[split:]

        # Copy
        for f in train_imgs:
            shutil.copy2(os.path.join(src_folder, f), os.path.join(train_folder, f))
        for f in val_imgs:
            shutil.copy2(os.path.join(src_folder, f), os.path.join(val_folder, f))

        print(f"{cls:<15} {len(images):>7} {len(train_imgs):>7} {len(val_imgs):>7}")

    print("-" * 40)
    print("เสร็จแล้ว! ✅")

if __name__ == "__main__":
    print("=" * 40)
    print("  Rebuild garbage_dataset_cls (80/20)")
    print("=" * 40)
    main()
