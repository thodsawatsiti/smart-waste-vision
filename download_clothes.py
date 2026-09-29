"""
download_clothes.py
Download clothes/fabric images from internet via Bing/Google
เพิ่มความหลากหลายให้ dataset clothes

วิธีใช้:
    python download_clothes.py
"""

import os
import sys
import random
import shutil
from icrawler.builtin import BingImageCrawler

sys.stdout.reconfigure(encoding='utf-8')

# ตำแหน่งปลายทาง
TARGET_DIR = r"C:\Users\popjr\Desktop\garbage\garbage-detection-yolov8\garbage-classification-v2\garbage-dataset\clothes"
TEMP_DIR = r"C:\Users\popjr\Desktop\garbage\garbage-detection-yolov8\downloaded_clothes_temp"
MAX_TOTAL = 1000  # จำกัด clothes ไม่เกิน 1000 รูป

# คำค้นหา (เน้นความหลากหลาย)
QUERIES = {
    'microfiber_cloth': 50,
    'old towel pile': 50,
    'rag cleaning cloth': 50,
    'fabric crumpled': 50,
    'old clothes pile': 80,
    'textile waste': 50,
    'blanket folded': 40,
    'bed sheet old': 40,
    'cleaning rag dirty': 50,
    'fuzzy fabric': 40,
}


def download_query(query, count, save_dir):
    """Download รูปจาก query"""
    print(f"\n  Downloading: '{query}' ({count} images)...")

    crawler = BingImageCrawler(
        storage={'root_dir': save_dir},
        downloader_threads=4,
    )

    try:
        crawler.crawl(
            keyword=query,
            max_num=count,
            min_size=(200, 200),
            file_idx_offset=0,
        )
        # นับไฟล์
        files = [f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        print(f"  Got {len(files)} images")
        return len(files)
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0


def main():
    print("=" * 60)
    print("  Download Clothes Images from Internet")
    print("=" * 60)

    # นับรูปเดิมก่อน
    if os.path.exists(TARGET_DIR):
        existing = [f for f in os.listdir(TARGET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\nClothes ปัจจุบัน: {len(existing)} รูป")
        next_num = len(existing) + 1
    else:
        os.makedirs(TARGET_DIR, exist_ok=True)
        next_num = 1
        print("\nClothes folder ว่าง — สร้างใหม่")

    # สร้าง temp folder
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    # Download แต่ละ query
    total_downloaded = 0
    for query, count in QUERIES.items():
        query_dir = os.path.join(TEMP_DIR, query.replace(' ', '_'))
        os.makedirs(query_dir, exist_ok=True)
        downloaded = download_query(query, count, query_dir)
        total_downloaded += downloaded

    print(f"\n{'=' * 60}")
    print(f"  Download เสร็จ — รวม {total_downloaded} รูป")
    print(f"{'=' * 60}")

    # ย้าย + เปลี่ยนชื่อรวมเข้า target
    print(f"\nกำลังย้ายเข้า {TARGET_DIR}...")
    new_files = []  # เก็บชื่อรูปที่ download มาใหม่
    for root, dirs, files in os.walk(TEMP_DIR):
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            src = os.path.join(root, fname)
            new_name = f"clothes_web_{next_num}.jpg"
            dst = os.path.join(TARGET_DIR, new_name)
            try:
                shutil.move(src, dst)
                new_files.append(new_name)
                next_num += 1
            except Exception:
                pass

    # ลบ temp
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print(f"  ย้ายเข้า dataset: {len(new_files)} รูป")

    # ตรวจสอบ + ลบ random ของเก่าให้รวมเหลือ MAX_TOTAL
    all_files = [f for f in os.listdir(TARGET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_now = len(all_files)
    print(f"\nรวมตอนนี้: {total_now} รูป (เป้าหมาย {MAX_TOTAL})")

    if total_now > MAX_TOTAL:
        # เก็บรูปใหม่ทั้งหมด + random ลบของเก่า
        new_set = set(new_files)
        old_files = [f for f in all_files if f not in new_set]
        excess = total_now - MAX_TOTAL
        to_remove = random.sample(old_files, min(excess, len(old_files)))

        print(f"\nกำลังลบรูปเก่า {len(to_remove)} รูป (เก็บรูปใหม่ {len(new_files)} ไว้ทั้งหมด)...")
        for f in to_remove:
            try:
                os.remove(os.path.join(TARGET_DIR, f))
            except Exception:
                pass

    final = [f for f in os.listdir(TARGET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    new_count = sum(1 for f in final if f.startswith('clothes_web_'))
    old_count = len(final) - new_count
    print(f"\n✅ เสร็จเรียบร้อย!")
    print(f"  Clothes ทั้งหมด: {len(final)} รูป")
    print(f"     ↳ รูปใหม่จาก internet: {new_count}")
    print(f"     ↳ รูปเก่าที่เก็บไว้: {old_count}")


if __name__ == "__main__":
    main()
