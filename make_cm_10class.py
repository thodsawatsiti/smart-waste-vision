"""
สร้าง confusion matrix แบบครบ 10 คลาส (รวม shoes) normalized
สำหรับ no_processing และ with_processing_v2_retrain
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sn
from ultralytics import YOLO

OUT_DIR = r"C:\Users\popjr\Downloads"
MODELS = [
    {"tag": "no_processing", "weights": "runs/classify/runs/garbage_cls_no_processing_v2/weights/best.pt", "data": "garbage_dataset_cls"},
    {"tag": "with_processing_v2_retrain", "weights": "runs/classify/runs/classify/runs/garbage_cls_with_processing_v2_retrain/weights/best.pt", "data": "garbage_dataset_cls_processed_v2"},
]


def plot_cm(matrix, names, save_path, normalize=True):
    array = matrix.copy().astype(float)
    if normalize:
        array = array / (array.sum(0).reshape(1, -1) + 1e-9)
        array[array < 0.005] = np.nan
        fmt, title = ".2f", "Confusion Matrix Normalized"
    else:
        array[array == 0] = np.nan
        fmt, title = ".0f", "Confusion Matrix"
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    sn.heatmap(array, ax=ax, annot=True, annot_kws={"size": 11}, cmap="Blues", fmt=fmt,
               square=True, vmin=0.0, xticklabels=names, yticklabels=names, cbar=True,
               linecolor="white", linewidths=0.5)
    ax.set_xlabel("True", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    fig.savefig(save_path, dpi=200); plt.close(fig)
    print(f"  saved: {save_path}")


for m in MODELS:
    print(f"\n=== {m['tag']} ===", flush=True)
    model = YOLO(m["weights"])
    results = model.val(data=m["data"], split="val", verbose=False, workers=0, device=0, batch=32)
    cm = results.confusion_matrix.matrix
    names = [model.names[i] for i in range(len(model.names))] + ["background"]
    # เอาเฉพาะ background ออก เก็บครบ 10 คลาส
    keep = [i for i in range(len(names)) if names[i] != "background"]
    cm2 = cm[np.ix_(keep, keep)]
    names2 = [names[i] for i in keep]
    plot_cm(cm2, names2, os.path.join(OUT_DIR, f"cm_{m['tag']}_10class_normalized.png"), normalize=True)
    plot_cm(cm2, names2, os.path.join(OUT_DIR, f"cm_{m['tag']}_10class_raw.png"), normalize=False)

print("\nDONE")
