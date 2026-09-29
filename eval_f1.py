"""
eval_f1.py
คำนวณ F1-score ของระบบ (โมเดลที่ดีที่สุด: no_processing_v1)
ใช้ model.val() เพื่อให้ตัวเลขตรงกับที่รายงาน (95.49%)

Output: ตาราง Precision/Recall/F1 รายคลาส + macro/weighted
        results/system_f1_report.csv
"""
import os, sys, io, csv
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ค่าเริ่มต้น = โมเดลที่ใช้งานจริงตอนนี้ (with_processing) + val set ที่ processed แล้ว
# แทนที่ได้ด้วย: python eval_f1.py <model.pt> <dataset_dir>
MODEL = sys.argv[1] if len(sys.argv) > 1 else "runs/classify/runs/garbage_cls_with_processing/weights/best.pt"
DATA = sys.argv[2] if len(sys.argv) > 2 else "garbage_dataset_cls_processed"
OUT = "results/system_f1_report.csv"


def plot_confusion(cm, classes, top1, out_png):
    """วาด confusion matrix เป็นรูปสำหรับใส่สไลด์"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    TH = fm.FontProperties(fname=r"C:\Windows\Fonts\tahoma.ttf")

    n = len(classes)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9.5, 8), dpi=200)
    fig.patch.set_facecolor("white")
    im = ax.imshow(cmn, cmap="Greens", vmin=0, vmax=1)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontproperties=TH, fontsize=11)
    ax.set_yticklabels(classes, fontproperties=TH, fontsize=11)
    ax.set_xlabel("ประเภทที่ระบบทำนาย (Predicted)", fontproperties=TH, fontsize=12,
                  fontweight="bold", labelpad=10)
    ax.set_ylabel("ประเภทที่ถูกต้อง (True)", fontproperties=TH, fontsize=12,
                  fontweight="bold", labelpad=10)
    ax.set_title(f"Confusion Matrix — Smart Waste Vision\nTop-1 Accuracy = {top1:.2f}%",
                 fontproperties=TH, fontsize=15, fontweight="bold", pad=16)

    for i in range(n):
        for j in range(n):
            v = int(cm[i][j])
            if v > 0:
                ax.text(j, i, v, ha="center", va="center", fontsize=9.5,
                        color="white" if cmn[i][j] > 0.55 else "#0f172a",
                        fontweight="bold" if i == j else "normal")
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.03)
    cbar.set_label("สัดส่วนต่อคลาส", fontproperties=TH, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close()
    print(f"  Saved: {out_png}")


def get_cm(metrics, model):
    """ดึง confusion matrix จาก validator"""
    for obj in (metrics, getattr(model, "validator", None)):
        cmobj = getattr(obj, "confusion_matrix", None)
        if cmobj is not None and hasattr(cmobj, "matrix"):
            return np.array(cmobj.matrix)
    return None


def main():
    os.makedirs("results", exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(MODEL)
    classes = [model.names[i] for i in range(len(model.names))]

    print("กำลังประเมินผลบน val set...\n", flush=True)
    metrics = model.val(data=DATA, split="val", verbose=False, plots=False)

    top1 = float(metrics.top1) * 100
    top5 = float(metrics.top5) * 100

    cm = get_cm(metrics, model)
    if cm is None:
        print("ไม่พบ confusion matrix"); return
    # ultralytics: matrix[pred][true] -> transpose ให้เป็น [true][pred]
    if cm.shape[0] == len(classes) + 1:
        cm = cm[:len(classes), :len(classes)]
    cm = cm.T.astype(float)

    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / np.maximum(tp + fn, 1e-9)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    sup = cm.sum(1)

    macro_f1 = f1.mean()
    macro_p, macro_r = prec.mean(), rec.mean()
    w = sup / sup.sum()
    wf1, wp, wr = (f1 * w).sum(), (prec * w).sum(), (rec * w).sum()

    print("=" * 68)
    print("  F1-SCORE REPORT — Smart Waste Vision (YOLOv11n-cls)")
    print("=" * 68)
    print(f"  Top-1 Accuracy: {top1:.2f}%   |   Top-5 Accuracy: {top5:.2f}%\n")
    print(f"{'Class':<14}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Support':<10}")
    print("-" * 68)
    for i, c in enumerate(classes):
        print(f"{c:<14}{prec[i]:<12.4f}{rec[i]:<12.4f}{f1[i]:<12.4f}{int(sup[i]):<10}")
    print("-" * 68)
    print(f"{'Macro Avg':<14}{macro_p:<12.4f}{macro_r:<12.4f}{macro_f1:<12.4f}{int(sup.sum()):<10}")
    print(f"{'Weighted Avg':<14}{wp:<12.4f}{wr:<12.4f}{wf1:<12.4f}{int(sup.sum()):<10}")
    print("=" * 68)
    print(f"\n  ** F1-Score ของระบบ = {wf1:.4f} (weighted) / {macro_f1:.4f} (macro) **")

    def write_csv(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["class", "precision", "recall", "f1_score", "support"])
            for i, c in enumerate(classes):
                wcsv.writerow([c, f"{prec[i]:.4f}", f"{rec[i]:.4f}", f"{f1[i]:.4f}", int(sup[i])])
            wcsv.writerow(["macro_avg", f"{macro_p:.4f}", f"{macro_r:.4f}", f"{macro_f1:.4f}", int(sup.sum())])
            wcsv.writerow(["weighted_avg", f"{wp:.4f}", f"{wr:.4f}", f"{wf1:.4f}", int(sup.sum())])
            wcsv.writerow(["top1_accuracy_%", f"{top1:.2f}", "", "", ""])
            wcsv.writerow(["top5_accuracy_%", f"{top5:.2f}", "", "", ""])

    try:
        write_csv(OUT)
        print(f"\n  Saved: {OUT}")
    except PermissionError:
        alt = OUT.replace(".csv", "_new.csv")
        try:
            write_csv(alt)
            print(f"\n  ⚠️  {OUT} ถูกเปิดค้างอยู่ (ปิด Excel ก่อน) — บันทึกเป็น {alt} แทน")
        except Exception as e:
            print(f"\n  ⚠️  บันทึก CSV ไม่ได้: {e}")

    plot_confusion(cm, classes, top1, "results/system_confusion_matrix.png")


if __name__ == "__main__":
    main()
