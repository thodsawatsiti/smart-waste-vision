"""
ab_clahe_sam.py
====================================================================
เปรียบเทียบ preprocessing: SAM  vs  CLAHE + SAM
เทรนบน data เดียวกัน (100 รูป/class) — คุมทุกอย่างเท่ากัน
วัดผล: Top-1, F1 (macro/weighted), Confusion Matrix

รัน:
    python ab_clahe_sam.py
ผลลัพธ์:
    results/ab_clahe_sam_summary.csv
    results/ab_clahe_sam_confusion.png
====================================================================
"""
import os, sys, io, glob, random, csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC      = "garbage_dataset_cls"
EXP      = "exp_clahe_sam"
N_TRAIN  = 100
N_VAL    = 30
PROC_SIZE = 320
IMGSZ    = 224
EPOCHS   = 20
SEED     = 42


# ---------- preprocess ----------
def clahe_color(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def make_procs():
    """คืน dict ของฟังก์ชัน process (โหลด FastSAM ครั้งเดียว)"""
    from ultralytics import FastSAM
    sam = FastSAM("FastSAM-s.pt")

    def sam_removebg(img):
        h, w = img.shape[:2]
        try:
            res = sam(img, device=0, retina_masks=True, imgsz=IMGSZ,
                      conf=0.4, iou=0.9, verbose=False)
            if res[0].masks is None:
                return img
            masks = res[0].masks.data.cpu().numpy()
            cy, cx = h // 2, w // 2
            best, ba = None, 0
            for m in masks:
                m = cv2.resize(m.astype("uint8"), (w, h), interpolation=cv2.INTER_NEAREST)
                area = m.sum() / (h * w)
                if 0.05 < area < 0.90 and m[cy, cx] == 1 and m.sum() > ba:
                    best, ba = m, m.sum()
            return img * best[:, :, None] if best is not None else img
        except Exception:
            return img

    return {
        "sam":       sam_removebg,
        "clahe_sam": lambda img: sam_removebg(clahe_color(img)),
    }


# ---------- สร้าง dataset ----------
def build_raw(classes):
    random.seed(SEED)
    for split, n in [("train", N_TRAIN), ("val", N_VAL)]:
        for cls in classes:
            files = glob.glob(os.path.join(SRC, split, cls, "*.jpg")) + \
                    glob.glob(os.path.join(SRC, split, cls, "*.png"))
            random.shuffle(files); files = files[:n]
            outdir = os.path.join(EXP, "raw", split, cls); os.makedirs(outdir, exist_ok=True)
            for i, f in enumerate(files):
                img = cv2.imread(f)
                if img is not None:
                    cv2.imwrite(os.path.join(outdir, f"{cls}_{i}.jpg"), img)


def build_processed(variant, proc_fn, classes):
    for split in ["train", "val"]:
        for cls in classes:
            src = os.path.join(EXP, "raw", split, cls)
            outdir = os.path.join(EXP, variant, split, cls); os.makedirs(outdir, exist_ok=True)
            for f in glob.glob(os.path.join(src, "*.jpg")):
                img = cv2.imread(f)
                if img is None: continue
                h, w = img.shape[:2]; s = PROC_SIZE / max(h, w)
                img = cv2.resize(img, (int(w*s), int(h*s)))
                out = proc_fn(img)
                cv2.imwrite(os.path.join(outdir, os.path.basename(f)), out if out is not None else img)


# ---------- เทรน + วัดผล ----------
def train_eval(variant, classes):
    from ultralytics import YOLO
    print(f"\n{'='*60}\n  เทรน: {variant}\n{'='*60}", flush=True)
    model = YOLO("yolo11n-cls.pt")
    model.train(data=os.path.join(EXP, variant), epochs=EPOCHS, imgsz=IMGSZ, batch=32,
                device=0, project="runs/classify/ab_clahe_sam", name=variant,
                exist_ok=True, pretrained=True, seed=SEED, cos_lr=True,
                verbose=False, plots=False, workers=0)
    name2idx = {v: k for k, v in model.names.items()}
    n = len(classes); cm = np.zeros((n, n), dtype=int)
    for cls in classes:
        ti = name2idx[cls]
        paths = glob.glob(os.path.join(EXP, variant, "val", cls, "*.jpg"))
        for p, res in zip(paths, model.predict(paths, verbose=False, stream=True, device=0)):
            cm[ti][int(res.probs.top1)] += 1
    tp = np.diag(cm).astype(float); fp = cm.sum(0)-tp; fn = cm.sum(1)-tp
    prec = tp/np.maximum(tp+fp, 1e-9); rec = tp/np.maximum(tp+fn, 1e-9)
    f1 = 2*prec*rec/np.maximum(prec+rec, 1e-9); sup = cm.sum(1)
    top1 = tp.sum()/cm.sum()*100
    macro_f1 = f1.mean(); wf1 = (f1*sup).sum()/sup.sum()
    print(f"  >>> {variant}: Top-1={top1:.2f}%  macro-F1={macro_f1:.4f}  weighted-F1={wf1:.4f}", flush=True)
    return dict(variant=variant, top1=top1, macro_f1=macro_f1, weighted_f1=wf1, cm=cm)


def plot_confusions(results, classes):
    fig, axes = plt.subplots(1, len(results), figsize=(7.5*len(results), 6.5))
    if len(results) == 1: axes = [axes]
    for ax, r in zip(axes, results):
        cm = r["cm"]; cmn = cm/cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{r['variant']}  (Top-1={r['top1']:.1f}%, F1={r['weighted_f1']:.3f})",
                     fontsize=12, fontweight="bold")
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i][j] > 0:
                    ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=7,
                            color="white" if cmn[i][j] > 0.5 else "#333")
    plt.suptitle("Confusion Matrix: SAM vs CLAHE + SAM", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/ab_clahe_sam_confusion.png", dpi=130, bbox_inches="tight")
    print("\n  Saved: results/ab_clahe_sam_confusion.png")


def main():
    os.makedirs("results", exist_ok=True)
    classes = sorted(d for d in os.listdir(os.path.join(SRC, "train"))
                     if os.path.isdir(os.path.join(SRC, "train", d)))
    print(f"คลาส: {len(classes)} | {N_TRAIN} train + {N_VAL} val ต่อ class")

    print("\n[1/3] สร้าง raw dataset...")
    build_raw(classes)
    procs = make_procs()
    print("[2/3] สร้าง SAM dataset...")
    build_processed("sam", procs["sam"], classes)
    print("[3/3] สร้าง CLAHE+SAM dataset...")
    build_processed("clahe_sam", procs["clahe_sam"], classes)

    results = [train_eval(v, classes) for v in ["sam", "clahe_sam"]]

    with open("results/ab_clahe_sam_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["variant", "top1_%", "macro_f1", "weighted_f1"])
        for r in results:
            w.writerow([r["variant"], f"{r['top1']:.2f}", f"{r['macro_f1']:.4f}", f"{r['weighted_f1']:.4f}"])

    print("\n" + "="*56)
    print("  สรุปผล: SAM vs CLAHE+SAM (100 รูป/class, 20 ep, seed=42)")
    print("="*56)
    print(f"{'Variant':<14}{'Top-1':<12}{'macro-F1':<12}{'weighted-F1':<12}")
    print("-"*50)
    for r in sorted(results, key=lambda x: -x["top1"]):
        print(f"{r['variant']:<14}{r['top1']:<12.2f}{r['macro_f1']:<12.4f}{r['weighted_f1']:<12.4f}")
    best = max(results, key=lambda x: x["top1"])
    print(f"\n  ชนะ: {best['variant']}  (Top-1={best['top1']:.2f}%)")
    plot_confusions(results, classes)
    print("  รายงาน: results/ab_clahe_sam_summary.csv")


if __name__ == "__main__":
    main()
