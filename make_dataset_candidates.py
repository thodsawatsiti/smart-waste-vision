"""
สร้างแผ่นตัวอย่างภาพ (contact sheet) แยกตามประเภท เพื่อให้เลือกภาพที่จะใช้ในรายงาน
ผลลัพธ์: C:\\Users\\popjr\\Downloads\\dataset_candidates\\<ประเภท>.png
เปิดดูแล้วจดชื่อไฟล์ใต้ภาพที่ชอบ ไปใส่ใน make_figure4_dataset.py

รัน:  python make_dataset_candidates.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

ROOT = r"C:\Users\popjr\Desktop\garbage\garbage-detection-yolov8\garbage_dataset_cls\train"
OUT = r"C:\Users\popjr\Downloads\dataset_candidates"
CLASSES = ["battery", "biological", "cardboard", "clothes", "glass",
           "metal", "paper", "plastic", "shoes", "trash"]
NCOL, NROW = 6, 4          # แสดง 24 ภาพต่อประเภท
os.makedirs(OUT, exist_ok=True)


def square(path, size=260):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = im.size
    s = min(w, h)
    return im.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2)).resize((size, size), Image.LANCZOS)


for cls in CLASSES:
    d = os.path.join(ROOT, cls)
    files = sorted(f for f in os.listdir(d) if not f.startswith("hard_"))
    # กระจายให้ทั่วชุดข้อมูล ไม่กระจุกอยู่ต้นๆ
    step = max(1, len(files) // (NCOL*NROW))
    pick = files[::step][:NCOL*NROW]

    fig, axes = plt.subplots(NROW, NCOL, figsize=(NCOL*2.1, NROW*2.35), dpi=100)
    fig.suptitle(cls, fontsize=13, y=0.995)
    for ax, f in zip(axes.flat, pick):
        try:
            ax.imshow(square(os.path.join(d, f)))
        except Exception:
            pass
        ax.set_title(f, fontsize=6.5, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(pick):]:
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    p = os.path.join(OUT, f"{cls}.png")
    plt.savefig(p, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {cls:11s} -> {p}")

print(f"\nเสร็จแล้ว เปิดดูที่โฟลเดอร์: {OUT}")
