"""
eval_deleaked.py
วัด accuracy จริงบน val หลังหักรูปที่ leak (ซ้ำกับ train) ออก
โมเดล: no_processing_v1 (ที่รายงาน 95.49%)
"""
import os, sys, io, glob
import numpy as np
from PIL import Image
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATASET = "garbage_dataset_cls"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "runs/classify/runs/garbage_cls_no_processing_v2/weights/best.pt"

def ahash(path, size=8):
    img = Image.open(path).convert("L").resize((size, size))
    px = np.asarray(img, dtype=np.float32)
    bits = (px > px.mean()).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h

def imgs(split, cls):
    d = os.path.join(DATASET, split, cls)
    return glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png"))

def main():
    classes = sorted(d for d in os.listdir(os.path.join(DATASET, "train"))
                     if os.path.isdir(os.path.join(DATASET, "train", d)))

    # 1. hash ของ train ทั้งหมด
    print("hashing train...", flush=True)
    train_hashes = set()
    for cls in classes:
        for p in imgs("train", cls):
            try: train_hashes.add(ahash(p))
            except: pass
    print(f"train hashes: {len(train_hashes)}", flush=True)

    # 2. predict val + เช็ค leak
    print("evaluating val...", flush=True)
    from ultralytics import YOLO
    model = YOLO(MODEL)
    name2idx = {v: k for k, v in model.names.items()}

    tot = cor = 0
    lk_tot = lk_cor = 0
    cl_tot = cl_cor = 0
    for cls in classes:
        paths = imgs("val", cls)
        ti = name2idx[cls]
        for p, res in zip(paths, model.predict(paths, stream=True, device=0, verbose=False)):
            ok = int(int(res.probs.top1) == ti)
            tot += 1; cor += ok
            try: leaked = ahash(p) in train_hashes
            except: leaked = False
            if leaked:
                lk_tot += 1; lk_cor += ok
            else:
                cl_tot += 1; cl_cor += ok

    def pct(a, b): return a / b * 100 if b else 0.0
    print("\n" + "=" * 60)
    print("  RESULT: accuracy จริงหลังหัก leakage")
    print("=" * 60)
    print(f"  val ทั้งหมด (มี leak)     : {pct(cor,tot):6.2f}%   ({cor}/{tot})   <- ที่รายงานเดิม")
    print(f"  รูปที่ leak (โมเดลเคยเห็น) : {pct(lk_cor,lk_tot):6.2f}%   ({lk_cor}/{lk_tot})")
    print(f"  ** val สะอาด (ไม่ leak) ** : {pct(cl_cor,cl_tot):6.2f}%   ({cl_cor}/{cl_tot})   <- เลขจริง")
    print(f"\n  ส่วนต่าง (inflation)      : {pct(cor,tot)-pct(cl_cor,cl_tot):+.2f}%")

if __name__ == "__main__":
    main()
