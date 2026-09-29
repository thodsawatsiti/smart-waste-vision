"""
ab3_raw_grabcut_sam.py
====================================================================
A/B 3 ทาง: raw vs GrabCut vs SAM — เทรนบน data เดียวกัน (100 รูป/class)
วัดผล: Top-1 Accuracy, F1-score (macro/weighted), Confusion Matrix

คุมทุกอย่างเท่ากัน เปลี่ยนแค่วิธี preprocess -> เทียบเป็นธรรม
รัน:
    python ab3_raw_grabcut_sam.py
ผลลัพธ์:
    results/ab3_summary.csv
    results/ab3_confusion.png   (confusion matrix 3 แบบเทียบกัน)
====================================================================
"""
import os, sys, io, glob, shutil, random
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---------- CONFIG ----------
SRC      = "garbage_dataset_cls"     # dataset ต้นทาง (raw)
EXP      = "exp_ab3"                 # โฟลเดอร์ dataset ชั่วคราว
N_TRAIN  = 100                       # รูป/class สำหรับ train
N_VAL    = 30                        # รูป/class สำหรับ val
PROC_SIZE = 320                      # ย่อก่อน process ให้เร็ว
IMGSZ    = 224
EPOCHS   = 20
SEED     = 42
VARIANTS = ["raw", "grabcut", "sam"]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------- ฟังก์ชัน preprocess ----------
def proc_grabcut(img):
    """CLAHE + GrabCut (ใช้ pipeline เดิมในโปรเจค)"""
    from image_processing import smart_preprocess
    try:
        out = smart_preprocess(img)
        return out if out is not None else img
    except Exception:
        return img


def make_sam_fn():
    """คืนฟังก์ชัน SAM (โหลด FastSAM ครั้งเดียว)"""
    from ultralytics import FastSAM
    sam = FastSAM("FastSAM-s.pt")

    def proc_sam(img):
        h, w = img.shape[:2]
        try:
            res = sam(img, device=0, retina_masks=True, imgsz=IMGSZ,
                      conf=0.4, iou=0.9, verbose=False)
            if res[0].masks is None:
                return img
            masks = res[0].masks.data.cpu().numpy()
            cy, cx = h // 2, w // 2
            best, best_area = None, 0
            for m in masks:
                m = cv2.resize(m.astype("uint8"), (w, h), interpolation=cv2.INTER_NEAREST)
                area = m.sum() / (h * w)
                if 0.05 < area < 0.90 and m[cy, cx] == 1 and m.sum() > best_area:
                    best, best_area = m, m.sum()
            if best is None:
                return img
            return img * best[:, :, None]
        except Exception:
            return img
    return proc_sam


# ---------- สร้าง dataset ----------
def build_raw(classes):
    random.seed(SEED)
    for split, n in [("train", N_TRAIN), ("val", N_VAL)]:
        for cls in classes:
            files = glob.glob(os.path.join(SRC, split, cls, "*.jpg")) + \
                    glob.glob(os.path.join(SRC, split, cls, "*.png"))
            random.shuffle(files)
            files = files[:n]
            outdir = os.path.join(EXP, "raw", split, cls)
            os.makedirs(outdir, exist_ok=True)
            for i, f in enumerate(files):
                img = cv2.imread(f)
                if img is None:
                    continue
                cv2.imwrite(os.path.join(outdir, f"{cls}_{i}.jpg"), img)


def build_processed(variant, proc_fn, classes):
    for split in ["train", "val"]:
        for cls in classes:
            src = os.path.join(EXP, "raw", split, cls)
            outdir = os.path.join(EXP, variant, split, cls)
            os.makedirs(outdir, exist_ok=True)
            for f in glob.glob(os.path.join(src, "*.jpg")):
                img = cv2.imread(f)
                if img is None:
                    continue
                # ย่อก่อน process ให้เร็ว
                h, w = img.shape[:2]
                s = PROC_SIZE / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)))
                out = proc_fn(img)
                if out is None:
                    out = img
                cv2.imwrite(os.path.join(outdir, os.path.basename(f)), out)


# ---------- เทรน + วัดผล ----------
def train_eval(variant, classes):
    from ultralytics import YOLO
    print(f"\n{'='*60}\n  เทรน: {variant}\n{'='*60}", flush=True)
    model = YOLO("yolo11n-cls.pt")
    model.train(data=os.path.join(EXP, variant), epochs=EPOCHS, imgsz=IMGSZ,
                batch=32, device=0, project="runs/classify/ab3", name=variant,
                exist_ok=True, pretrained=True, seed=SEED, cos_lr=True,
                verbose=False, plots=False, workers=0)

    # predict บน val -> เก็บ true/pred
    name2idx = {v: k for k, v in model.names.items()}
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)   # cm[true][pred]
    for cls in classes:
        ti = name2idx[cls]
        paths = glob.glob(os.path.join(EXP, variant, "val", cls, "*.jpg"))
        for p, res in zip(paths, model.predict(paths, verbose=False, stream=True, device=0)):
            pi = int(res.probs.top1)
            cm[ti][pi] += 1

    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    support = cm.sum(1)
    top1 = tp.sum() / cm.sum() * 100
    macro_f1 = f1.mean()
    weighted_f1 = (f1 * support).sum() / support.sum()
    print(f"  >>> {variant}: Top-1={top1:.2f}%  macro-F1={macro_f1:.4f}  weighted-F1={weighted_f1:.4f}", flush=True)
    return dict(variant=variant, top1=top1, macro_f1=macro_f1,
                weighted_f1=weighted_f1, cm=cm)


def plot_confusions(results, classes):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    for ax, r in zip(axes, results):
        cm = r["cm"]
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
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
    plt.suptitle("Confusion Matrix: raw vs GrabCut vs SAM", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/ab3_confusion.png", dpi=130, bbox_inches="tight")
    print("\n  Saved: results/ab3_confusion.png")


def main():
    os.makedirs("results", exist_ok=True)
    classes = sorted(d for d in os.listdir(os.path.join(SRC, "train"))
                     if os.path.isdir(os.path.join(SRC, "train", d)))
    print(f"คลาส: {len(classes)} | {N_TRAIN} train + {N_VAL} val ต่อ class")

    print("\n[1/3] สร้าง raw dataset...")
    build_raw(classes)
    print("[2/3] สร้าง GrabCut dataset... (ช้าหน่อย)")
    build_processed("grabcut", proc_grabcut, classes)
    print("[3/3] สร้าง SAM dataset... (โหลด FastSAM ~23MB)")
    build_processed("sam", make_sam_fn(), classes)

    results = [train_eval(v, classes) for v in VARIANTS]

    # สรุป + บันทึก
    import csv
    with open("results/ab3_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "top1_%", "macro_f1", "weighted_f1"])
        for r in results:
            w.writerow([r["variant"], f"{r['top1']:.2f}", f"{r['macro_f1']:.4f}", f"{r['weighted_f1']:.4f}"])

    print("\n" + "=" * 60)
    print("  สรุปผล A/B 3 ทาง (100 รูป/class, 20 epochs, seed=42)")
    print("=" * 60)
    print(f"{'Variant':<12}{'Top-1':<12}{'macro-F1':<12}{'weighted-F1':<12}")
    print("-" * 48)
    for r in sorted(results, key=lambda x: -x["top1"]):
        print(f"{r['variant']:<12}{r['top1']:<12.2f}{r['macro_f1']:<12.4f}{r['weighted_f1']:<12.4f}")
    best = max(results, key=lambda x: x["top1"])
    print(f"\n  ชนะ: {best['variant']}  (Top-1={best['top1']:.2f}%)")

    plot_confusions(results, classes)
    print("\n  รายงาน: results/ab3_summary.csv + results/ab3_confusion.png")


if __name__ == "__main__":
    main()
