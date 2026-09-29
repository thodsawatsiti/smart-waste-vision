"""
preprocess_dataset_v2.py
ใช้ smart_preprocess() (เวอร์ชันไม่มี fallback แล้ว) กับ garbage_dataset_cls ปัจจุบัน
ซึ่งรวม hard examples จาก feedback loop (uploads/) เข้าไปแล้ว — ตามที่อาจารย์แนะนำ:
- CLAHE: ทุกรูป
- GrabCut: ทุกรูป เสมอ (ไม่มี fallback ตามคุณภาพ mask แล้ว กัน background ปนสไตล์กัน)
- ข้อมูลต้นทาง: เดียวกับที่ใช้เทรน no_processing (รวม hard_* ที่เพิ่มไปแล้ว)

วิธีใช้:
    python preprocess_dataset_v2.py

ความเร็ว:
- Multiprocessing: ~5-10 นาที (ใช้ทุก CPU core)
"""

import os
import sys
import cv2
import time
import multiprocessing as mp
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')


SOURCE = r"C:\Users\popjr\Desktop\garbage\garbage-detection-yolov8\garbage_dataset_cls"
TARGET = r"C:\Users\popjr\Desktop\garbage\garbage-detection-yolov8\garbage_dataset_cls_processed_v2"

# ใช้ CPU cores ทั้งหมด (เหลือ 1 core ให้ระบบ)
NUM_WORKERS = max(1, mp.cpu_count() - 1)


def process_single_image(args):
    """ Process รูปเดียว - ฟังก์ชันสำหรับ worker """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from image_processing import smart_preprocess

    src_path, dst_path = args
    try:
        if os.path.exists(dst_path):
            return ('skip', src_path)
        processed = smart_preprocess(src_path)
        if processed is not None:
            cv2.imwrite(dst_path, processed)
            return ('ok', src_path)
        else:
            return ('fail', src_path)
    except Exception as e:
        return ('error', f"{src_path}: {e}")


def collect_tasks():
    """รวบรวมทุก task (src, dst) ทั้ง train + val"""
    tasks = []
    splits = ['train', 'val']

    for split in splits:
        src_split = os.path.join(SOURCE, split)
        dst_split = os.path.join(TARGET, split)

        if not os.path.exists(src_split):
            continue

        classes = sorted([d for d in os.listdir(src_split) if os.path.isdir(os.path.join(src_split, d))])

        for cls in classes:
            src_cls = os.path.join(src_split, cls)
            dst_cls = os.path.join(dst_split, cls)
            os.makedirs(dst_cls, exist_ok=True)

            images = [f for f in os.listdir(src_cls) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for fname in images:
                tasks.append((os.path.join(src_cls, fname), os.path.join(dst_cls, fname)))

    return tasks


def main():
    print("=" * 60)
    print("  Preprocess Dataset v2 (CLAHE + GrabCut, no fallback)")
    print(f"  Source: {SOURCE}  (รวม hard examples แล้ว)")
    print(f"  Workers: {NUM_WORKERS} CPU cores")
    print("=" * 60)

    print("\n[1] รวบรวมรายการรูป...")
    tasks = collect_tasks()
    total = len(tasks)
    print(f"  พบ {total:,} รูป ที่ต้อง process")

    if total == 0:
        print("ไม่พบรูป")
        return

    print(f"\n[2] เริ่ม processing ด้วย {NUM_WORKERS} workers...")
    start = time.time()

    results = {'ok': 0, 'skip': 0, 'fail': 0, 'error': 0}
    errors = []

    with mp.Pool(NUM_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_image, tasks, chunksize=10), 1):
            status, info = result
            results[status] += 1
            if status == 'error':
                errors.append(info)

            if i % 200 == 0 or i == total:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                pct = i / total * 100
                print(f"  [{pct:5.1f}%] {i:>5}/{total} | "
                      f"OK={results['ok']:>4} skip={results['skip']:>4} fail={results['fail']:>3} | "
                      f"{rate:.1f}/s | ETA {eta/60:.1f}m")

    total_time = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"เสร็จสิ้น!")
    print(f"  OK:     {results['ok']:,}")
    print(f"  Skip:   {results['skip']:,}")
    print(f"  Fail:   {results['fail']:,}")
    print(f"  Error:  {results['error']:,}")
    print(f"  เวลา:   {total_time/60:.1f} นาที")
    print(f"  Output: {TARGET}")
    print(f"{'=' * 60}")

    if errors:
        print(f"\n⚠️  Errors ({len(errors)}):")
        for e in errors[:5]:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
