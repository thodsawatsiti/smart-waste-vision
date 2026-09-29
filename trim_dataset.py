"""
trim_dataset.py
ตัดรูปที่เกิน 1500 ต่อ class ออกแบบ random
"""

import os
import random

DATASET_PATH = r"C:\Users\popjr\Desktop\garbage\garbage-detection-yolov8\garbage-classification-v2\garbage-dataset"
MAX_PER_CLASS = 1000
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

def count_images(folder):
    return [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in EXTENSIONS]

def main():
    classes = sorted(os.listdir(DATASET_PATH))
    print(f"{'Class':<15} {'Before':>8} {'After':>8} {'Removed':>8}")
    print("-" * 45)

    total_removed = 0

    for cls in classes:
        folder = os.path.join(DATASET_PATH, cls)
        if not os.path.isdir(folder):
            continue

        images = count_images(folder)
        count = len(images)

        if count <= MAX_PER_CLASS:
            print(f"{cls:<15} {count:>8} {count:>8} {'':>8}  ✅")
            continue

        # Random เลือกรูปที่จะลบ
        to_remove = random.sample(images, count - MAX_PER_CLASS)
        for fname in to_remove:
            os.remove(os.path.join(folder, fname))

        removed = len(to_remove)
        total_removed += removed
        print(f"{cls:<15} {count:>8} {MAX_PER_CLASS:>8} {removed:>8}  ✂️")

    print("-" * 45)
    print(f"{'Total removed':<15} {total_removed:>8} รูป")
    print("\nเสร็จแล้ว! ✅")

if __name__ == "__main__":
    print("=" * 45)
    print("  Trim Dataset to 1500 images per class")
    print("=" * 45)
    confirm = input("\nยืนยันลบรูปถาวร? (yes/no): ").strip().lower()
    if confirm == "yes":
        main()
    else:
        print("ยกเลิก")
