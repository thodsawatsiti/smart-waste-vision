"""
add_hard_examples.py
ดึงรูปที่ user ยืนยันผ่าน feedback (ใน uploads/ + PostgreSQL) มาเสริมเข้า training set

Match ไฟล์ด้วย hash ต่อท้ายชื่อไฟล์ (ไม่ใช่ชื่อเต็ม) เพราะไฟล์ถูก rename ให้ตรงกับ
label จริงไปแล้ว ส่วน DB เก็บชื่อไฟล์ตอน upload ครั้งแรก (ก่อน feedback) ไว้เฉยๆ

แบ่งข้อมูลเป็น 2 ส่วน:
  - ส่วนใหญ่ (80%) -> oversample เข้า train/<class>/ เพื่อ fine-tune
  - ส่วนที่เหลือ (20%) -> เก็บแยกไว้ที่ hard_val/<class>/ เป็น holdout
    ไม่แตะต้อง ไม่เอาไปเทรน ใช้เช็คว่าโมเดล generalize จริงไหมหลัง fine-tune
"""
import os, shutil, sys, io, glob, re, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from database import SessionLocal, Prediction

UPLOAD_DIR = "uploads"
DATASET = "garbage_dataset_cls"
HARD_VAL_DIR = "hard_val"
OVERSAMPLE = 3
HOLDOUT_FRAC = 0.2
random.seed(42)

def suffix_of(fname):
    base, ext = os.path.splitext(fname)
    m = re.search(r"_([0-9a-fA-F]{6,10})$", base)
    return (m.group(1), ext) if m else (None, ext)

def main():
    db = SessionLocal()
    rows = db.query(Prediction).filter(Prediction.image_filename.isnot(None)).all()

    by_class = {}  # true_label -> [file_path, ...]
    no_file = 0
    for r in rows:
        if r.is_correct == "incorrect" and r.correct_class:
            true_label = r.correct_class
        elif r.is_correct == "correct":
            true_label = r.class_name
        else:
            continue

        h, ext = suffix_of(r.image_filename)
        if h is None:
            no_file += 1
            continue
        candidates = list(set(glob.glob(os.path.join(UPLOAD_DIR, f"*_{h}.*"))))
        if not candidates:
            no_file += 1
            continue
        by_class.setdefault(true_label, []).append(candidates[0])

    print(f"หาไฟล์ไม่เจอ/ไม่มี feedback ที่ใช้ได้: {no_file}\n")

    total_train, total_holdout = 0, 0
    for cls, paths in sorted(by_class.items(), key=lambda x: -len(x[1])):
        paths = sorted(set(paths))
        random.shuffle(paths)

        dst_dir = os.path.join(DATASET, "train", cls)
        if not os.path.isdir(dst_dir):
            print(f"  [!] ไม่มี class folder train/{cls} ข้ามทั้งหมด ({len(paths)} รูป)")
            continue

        n_holdout = max(1, int(len(paths) * HOLDOUT_FRAC)) if len(paths) >= 5 else 0
        holdout_paths = paths[:n_holdout]
        train_paths = paths[n_holdout:]

        # ส่วน train -> oversample
        for src in train_paths:
            base = os.path.splitext(os.path.basename(src))[0]
            ext = os.path.splitext(src)[1]
            for i in range(OVERSAMPLE):
                dst = os.path.join(dst_dir, f"hard_{base}_{i}{ext}")
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
        total_train += len(train_paths)

        # ส่วน holdout -> เก็บแยก ไม่ยุ่งกับ train
        if holdout_paths:
            hv_dir = os.path.join(HARD_VAL_DIR, cls)
            os.makedirs(hv_dir, exist_ok=True)
            for src in holdout_paths:
                dst = os.path.join(hv_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
        total_holdout += len(holdout_paths)

        print(f"  {cls}: {len(paths)} รูปจริง -> train {len(train_paths)} (x{OVERSAMPLE}), holdout {len(holdout_paths)}")

    print(f"\nรวม: train +{total_train} รูปต้นฉบับ (oversample x{OVERSAMPLE}), holdout {total_holdout} รูป (เก็บที่ {HARD_VAL_DIR}/)")

if __name__ == "__main__":
    main()
