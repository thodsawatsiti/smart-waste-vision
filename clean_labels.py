"""
clean_labels.py
ตรวจคุณภาพ dataset (garbage_dataset_cls) โดยไม่ลบอะไรเอง — ออกรายงานให้ตรวจ

ตรวจ 3 อย่าง:
1. รูปซ้ำ/เกือบซ้ำ (average hash) — เน้นตัวที่ซ้ำข้าม train<->val (data leakage)
2. label น่าสงสัย — โมเดลมั่นใจสูงว่าเป็นคลาสอื่น (val เชื่อถือได้กว่าเพราะไม่ได้เทรน)
3. รูปเสีย / เล็กเกินไป

Output: results/label_issues.csv + สรุปหน้าจอ
"""
import os, sys, io, glob, csv
from collections import defaultdict
import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATASET = "garbage_dataset_cls"
MODEL = "runs/classify/runs/garbage_cls_no_processing_v1/weights/best.pt"
OUT = "results/label_issues.csv"
CONF_FLAG = 0.85          # มั่นใจเกินนี้ + คลาสไม่ตรง = น่าสงสัย
MIN_SIZE = 32             # เล็กกว่านี้ = น่าสงสัย
os.makedirs("results", exist_ok=True)

# ---------- 1. รวบรวมรูปทั้งหมด ----------
items = []   # (split, cls, path)
for split in ["train", "val"]:
    base = os.path.join(DATASET, split)
    if not os.path.isdir(base):
        continue
    for cls in sorted(os.listdir(base)):
        cdir = os.path.join(base, cls)
        if not os.path.isdir(cdir):
            continue
        for p in glob.glob(os.path.join(cdir, "*.jpg")) + glob.glob(os.path.join(cdir, "*.png")):
            items.append((split, cls, p))
print(f"รูปทั้งหมด: {len(items)}")

issues = []   # (issue_type, detail, path)

# ---------- helper: average hash ----------
def ahash(path, size=8):
    img = Image.open(path).convert("L").resize((size, size))
    px = np.asarray(img, dtype=np.float32)
    bits = (px > px.mean()).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h

# ---------- 3. รูปเสีย/เล็ก + เก็บ hash ----------
print("กำลังตรวจรูปเสีย + คำนวณ hash...")
hash_map = defaultdict(list)   # hash -> [(split,cls,path)]
for i, (split, cls, p) in enumerate(items):
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            w, h = im.size
        if min(w, h) < MIN_SIZE:
            issues.append(("tiny_image", f"{w}x{h}px", p))
            continue
        hash_map[ahash(p)].append((split, cls, p))
    except Exception as e:
        issues.append(("corrupt_image", type(e).__name__, p))
    if (i + 1) % 2000 == 0:
        print(f"  ...{i+1}/{len(items)}")

# ---------- 1. รูปซ้ำ ----------
print("กำลังหารูปซ้ำ...")
dup_groups = 0
leak_groups = 0
for h, group in hash_map.items():
    if len(group) < 2:
        continue
    dup_groups += 1
    splits = set(g[0] for g in group)
    classes = set(g[1] for g in group)
    cross_split = len(splits) > 1
    if cross_split:
        leak_groups += 1
    tag = "DUP_LEAK(train<->val)" if cross_split else "DUP"
    if len(classes) > 1:
        tag += "+diff_class"
    detail = f"{tag} x{len(group)} | " + " ; ".join(f"{s}/{c}/{os.path.basename(p)}" for s, c, p in group)
    issues.append(("duplicate", detail, group[0][2]))

# ---------- 2. label น่าสงสัย ----------
print("กำลังตรวจ label น่าสงสัย (โหลดโมเดล)...")
suspect = 0
if os.path.exists(MODEL):
    from ultralytics import YOLO
    model = YOLO(MODEL)
    names = model.names
    by_folder = defaultdict(list)
    for split, cls, p in items:
        by_folder[(split, cls)].append(p)
    for (split, cls), paths in by_folder.items():
        for p, res in zip(paths, model.predict(paths, verbose=False, stream=True, device=0)):
            pi = int(res.probs.top1); conf = float(res.probs.top1conf)
            pred = names[pi]
            if pred != cls and conf >= CONF_FLAG:
                issues.append(("suspect_label",
                               f"folder={cls} แต่ทายเป็น {pred} ({conf:.2f}) [{split}]", p))
                suspect += 1
else:
    print(f"  ข้าม (ไม่พบโมเดล {MODEL})")

# ---------- บันทึก ----------
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["issue_type", "detail", "path"])
    for row in issues:
        w.writerow(row)

# ---------- สรุป ----------
print("\n" + "=" * 60)
print("  สรุปผลตรวจคุณภาพ dataset")
print("=" * 60)
n_dup = sum(1 for i in issues if i[0] == "duplicate")
n_corrupt = sum(1 for i in issues if i[0] == "corrupt_image")
n_tiny = sum(1 for i in issues if i[0] == "tiny_image")
print(f"  รูปทั้งหมด            : {len(items)}")
print(f"  กลุ่มรูปซ้ำ           : {dup_groups}  (ในนี้ซ้ำข้าม train<->val = {leak_groups} ⚠️)")
print(f"  label น่าสงสัย        : {suspect}")
print(f"  รูปเสีย               : {n_corrupt}")
print(f"  รูปเล็กเกินไป         : {n_tiny}")
print(f"\n  รายงานเต็ม: {OUT}")
if leak_groups:
    print(f"\n  ⚠️  พบรูปซ้ำข้าม train/val {leak_groups} กลุ่ม = data leakage")
    print(f"      -> accuracy จริงอาจต่ำกว่าที่รายงานเล็กน้อย")
