"""
validate_processed.py
ตรวจสอบคุณภาพรูปใน garbage_dataset_cls_processed/

เช็ค:
1. รูปอ่านได้ไหม (ไม่ corrupted)
2. ขนาดถูกต้อง
3. ไม่ขาวเกือบหมด (GrabCut ตัดเยอะเกิน)
4. ไม่ดำเกือบหมด

วิธีใช้:
    python validate_processed.py
"""

import os
import sys
import cv2
import numpy as np
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

PROCESSED_DIR = r"C:\Users\popjr\Desktop\garbage\garbage-detection-yolov8\garbage_dataset_cls_processed"

# Threshold สำหรับเช็ค
WHITE_THRESHOLD = 0.95   # ถ้าขาว > 95% ของรูป → ผิด
BLACK_THRESHOLD = 0.95   # ถ้าดำ > 95% → ผิด
MIN_SIZE_KB = 5          # ถ้าไฟล์เล็กกว่า 5KB → น่าสงสัย


def analyze_image(path):
    """ตรวจสอบรูป 1 รูป — return (status, reason)"""
    # เช็คขนาดไฟล์
    size_kb = os.path.getsize(path) / 1024
    if size_kb < MIN_SIZE_KB:
        return ('bad', f'file too small ({size_kb:.1f}KB)')

    # อ่านรูป
    img = cv2.imread(path)
    if img is None:
        return ('bad', 'cannot read')

    # เช็คขนาด
    h, w = img.shape[:2]
    if h < 100 or w < 100:
        return ('bad', f'too small ({w}x{h})')

    # เช็ค % ขาว/ดำ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_ratio = (gray > 240).sum() / gray.size
    black_ratio = (gray < 15).sum() / gray.size

    if white_ratio > WHITE_THRESHOLD:
        return ('warning', f'mostly white ({white_ratio*100:.0f}%)')

    if black_ratio > BLACK_THRESHOLD:
        return ('bad', f'mostly black ({black_ratio*100:.0f}%)')

    return ('ok', None)


def main():
    print("=" * 70)
    print("  ตรวจสอบคุณภาพรูปใน processed dataset")
    print("=" * 70)

    if not os.path.exists(PROCESSED_DIR):
        print(f"ไม่พบ folder: {PROCESSED_DIR}")
        print("กรุณารัน 'python preprocess_dataset.py' ก่อน")
        return

    splits = ['train', 'val']
    total_stats = {'ok': 0, 'warning': 0, 'bad': 0}
    bad_files = []
    warning_files = []

    for split in splits:
        split_path = os.path.join(PROCESSED_DIR, split)
        if not os.path.exists(split_path):
            continue

        classes = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])

        print(f"\n[{split.upper()}]")
        print(f"  {'Class':<15} {'OK':>5} {'Warning':>8} {'Bad':>5} {'Total':>6}")
        print(f"  {'-' * 50}")

        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            stats = {'ok': 0, 'warning': 0, 'bad': 0}

            for fname in images:
                fpath = os.path.join(cls_path, fname)
                status, reason = analyze_image(fpath)
                stats[status] += 1

                if status == 'bad':
                    bad_files.append((fpath, reason))
                elif status == 'warning':
                    warning_files.append((fpath, reason))

            total = sum(stats.values())
            print(f"  {cls:<15} {stats['ok']:>5} {stats['warning']:>8} {stats['bad']:>5} {total:>6}")

            for k in stats:
                total_stats[k] += stats[k]

    # Summary
    print(f"\n{'=' * 70}")
    print(f"📊 สรุปทั้งหมด:")
    grand_total = sum(total_stats.values())
    print(f"  ✅ OK:      {total_stats['ok']:>6,} ({total_stats['ok']/grand_total*100:.1f}%)")
    print(f"  ⚠️  Warning: {total_stats['warning']:>6,} ({total_stats['warning']/grand_total*100:.1f}%)")
    print(f"  ❌ Bad:     {total_stats['bad']:>6,} ({total_stats['bad']/grand_total*100:.1f}%)")
    print(f"  รวม:        {grand_total:>6,}")
    print(f"{'=' * 70}")

    # แสดงตัวอย่าง warning/bad
    if warning_files:
        print(f"\n⚠️  Warning files (แสดง 10 รูปแรก จาก {len(warning_files)}):")
        for fpath, reason in warning_files[:10]:
            rel = os.path.relpath(fpath, PROCESSED_DIR)
            print(f"  - {rel}: {reason}")

    if bad_files:
        print(f"\n❌ Bad files (แสดง 10 รูปแรก จาก {len(bad_files)}):")
        for fpath, reason in bad_files[:10]:
            rel = os.path.relpath(fpath, PROCESSED_DIR)
            print(f"  - {rel}: {reason}")

        # ถามว่าจะลบ bad files ไหม
        print(f"\n💡 พบ {len(bad_files)} รูปที่ใช้ train ไม่ได้")
        print("   รัน 'python validate_processed.py --delete' เพื่อลบทิ้งอัตโนมัติ")

    # Delete mode
    if '--delete' in sys.argv and bad_files:
        print(f"\n🗑️  กำลังลบ {len(bad_files)} bad files...")
        for fpath, _ in bad_files:
            try:
                os.remove(fpath)
            except Exception:
                pass
        print(f"  ลบเสร็จแล้ว ✅")


if __name__ == "__main__":
    main()
