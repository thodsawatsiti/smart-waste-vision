"""
download_new_data.py
โหลดรูปเพิ่มสำหรับคลาสอ่อน: ผ้าขี้ริ้ว (clothes) + กระป๋อง (metal)
เก็บไว้ใน new_data/ (review folder) — ยังไม่ใส่เข้า training set

เน้นคำค้นหลากหลาย + สภาพจริง (ไม่ใช่รูปสตูดิโอสะอาดๆ) เพื่อลด domain bias
"""
import os, sys, io, shutil, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from icrawler.builtin import BingImageCrawler

OUT = "new_data"

# คำค้น (ผสมไทย+อังกฤษ, เน้นสภาพจริง/หลากหลาย)
QUERIES = {
    "rag_clothes": [
        "ผ้าขี้ริ้ว", "ผ้าเช็ดโต๊ะเก่า", "ผ้าขี้ริ้วเก่า",
        "old cleaning rag cloth", "dish rag dirty", "microfiber cleaning cloth used",
        "rag pile textile waste", "worn out towel old",
    ],
    "can_metal": [
        "กระป๋องน้ำอัดลม", "กระป๋องบุบ", "กระป๋องเปล่า",
        "crushed aluminum soda can", "empty tin can food", "used beverage can trash",
        "dented metal can", "aluminum can recycling",
    ],
}
PER_QUERY = 30   # ~240 รูป/คลาส ก่อน dedupe


def download(cat, queries):
    base = os.path.join(OUT, cat)
    os.makedirs(base, exist_ok=True)
    for q in queries:
        qdir = os.path.join(base, "_tmp_" + q.replace(" ", "_")[:20])
        os.makedirs(qdir, exist_ok=True)
        print(f"\n  [{cat}] '{q}' ...", flush=True)
        try:
            c = BingImageCrawler(storage={"root_dir": qdir}, downloader_threads=4)
            c.crawl(keyword=q, max_num=PER_QUERY, min_size=(200, 200))
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
    # รวมไฟล์ + เปลี่ยนชื่อไม่ซ้ำ
    n = 1
    for root, _, files in os.walk(base):
        if root == base:
            continue
        for fn in files:
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                ext = ".jpg"
                dst = os.path.join(base, f"{cat}_web_{n}{ext}")
                try:
                    shutil.move(os.path.join(root, fn), dst)
                    n += 1
                except Exception:
                    pass
    # ลบ tmp
    for d in glob.glob(os.path.join(base, "_tmp_*")):
        shutil.rmtree(d, ignore_errors=True)
    total = len([f for f in os.listdir(base) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    return total


if __name__ == "__main__":
    print("=" * 60)
    print("  โหลดรูปเพิ่ม: ผ้าขี้ริ้ว + กระป๋อง")
    print("=" * 60)
    results = {}
    for cat, qs in QUERIES.items():
        results[cat] = download(cat, qs)
    print("\n" + "=" * 60)
    print("  สรุป (เก็บใน new_data/ — ยังไม่เข้า training set)")
    print("=" * 60)
    for cat, n in results.items():
        print(f"  {cat:<14}: {n} รูป")
    print(f"\n  ตรวจรูปแล้วค่อยย้ายเข้า dataset:")
    print(f"    new_data/rag_clothes/  -> garbage_dataset_cls/train/clothes/")
    print(f"    new_data/can_metal/    -> garbage_dataset_cls/train/metal/")
