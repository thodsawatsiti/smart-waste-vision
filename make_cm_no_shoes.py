"""
สร้าง confusion matrix ที่ตัดคลาส shoes ออก
สำหรับ 2 โมเดล: no_processing_v2 และ with_processing_v2_retrain
ทั้งแบบ normalized และ raw count  => รวม 4 รูป

วิธี: รัน model.val() ดึง confusion_matrix.matrix (nc+1 x nc+1)
แล้วลบแถว/คอลัมน์ของ shoes ออก ก่อน plot ใหม่ด้วย seaborn
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
    {
        "tag": "with_processing_v2_retrain",
        "weights": "runs/classify/runs/classify/runs/garbage_cls_with_processing_v2_retrain/weights/best.pt",
        "data": "garbage_dataset_cls_processed_v2",
    },
]


def plot_cm(matrix, names, normalize, title, save_path):
    """วาด confusion matrix สไตล์เดียวกับ ultralytics (Predicted=y, True=x, Blues)"""
    array = matrix.copy().astype(float)
    if normalize:
        array = array / (array.sum(0).reshape(1, -1) + 1e-9)  # normalize ต่อคอลัมน์ (true class)
        array[array < 0.005] = np.nan  # ช่องที่น้อยมาก เว้นว่าง (เหมือน ultralytics)
        fmt = ".2f"
    else:
        array[array == 0] = np.nan  # ช่องที่เป็น 0 เว้นว่าง
        fmt = ".0f"

    n = len(names)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    sn.heatmap(
        array,
        ax=ax,
        annot=True,
        annot_kws={"size": 11},
        cmap="Blues",
        fmt=fmt,
        square=True,
        vmin=0.0,
        xticklabels=names,
        yticklabels=names,
        cbar=True,
        linecolor="white",
        linewidths=0.5,
    )
    ax.set_xlabel("True", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  saved: {save_path}")


for m in MODELS:
    print(f"\n=== {m['tag']} ===", flush=True)
    model = YOLO(m["weights"])
    results = model.val(data=m["data"], split="val", verbose=False,
                        workers=0, device=0, batch=32)  # workers=0 กัน DataLoader ค้างบน Windows

    cm = results.confusion_matrix.matrix  # shape (nc+1, nc+1) รวม background
    # ชื่อคลาสตามลำดับ index + background ท้ายสุด
    names = [model.names[i] for i in range(len(model.names))] + ["background"]

    # หา index ของ shoes แล้วลบทั้งแถวและคอลัมน์
    shoes_idx = names.index("shoes")
    keep = [i for i in range(len(names)) if i != shoes_idx]
    cm2 = cm[np.ix_(keep, keep)]
    names2 = [names[i] for i in keep]

    plot_cm(cm2, names2, normalize=True,
            title="Confusion Matrix Normalized",
            save_path=os.path.join(OUT_DIR, f"cm_{m['tag']}_noshoes_normalized.png"))
    plot_cm(cm2, names2, normalize=False,
            title="Confusion Matrix",
            save_path=os.path.join(OUT_DIR, f"cm_{m['tag']}_noshoes_raw.png"))

print("\nDONE")
