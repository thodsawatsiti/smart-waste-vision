"""
ภาพที่ 4: ตัวอย่างภาพในชุดข้อมูลทั้ง 10 ประเภท (5 คอลัมน์ x 2 แถว)

วิธีเลือกภาพเอง:
  1) รัน  python make_dataset_candidates.py  เพื่อดูตัวอย่างภาพแต่ละประเภท
     (ผลลัพธ์อยู่ที่ C:\\Users\\popjr\\Downloads\\dataset_candidates\\)
  2) จดชื่อไฟล์ที่ชอบ มาใส่ใน PICK ด้านล่าง
     - ใส่แค่ชื่อไฟล์  เช่น  "glass": "glass_123.jpg"     (หาในโฟลเดอร์ train ของประเภทนั้น)
     - หรือใส่ path เต็ม เช่น "glass": r"C:\\Users\\popjr\\Desktop\\...\\uploads\\myphoto.jpg"
       (ใช้รูปที่ถ่ายเองก็ได้)
  3) รัน  python make_figure4_dataset.py

รัน:  python make_figure4_dataset.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageOps

TH_B = fm.FontProperties(fname=r"C:\Windows\Fonts\tahomabd.ttf")

ROOT = r"C:\Users\popjr\Desktop\garbage\garbage-detection-yolov8\garbage_dataset_cls\train"
OUTFILE = r"C:\Users\popjr\Downloads\figure4_dataset_samples.png"

TXT = "#14532d"
EDGE = "#86efac"

CLASSES = ["battery", "biological", "cardboard", "clothes", "glass",
           "metal", "paper", "plastic", "shoes", "trash"]

# ===== แก้ตรงนี้เพื่อเลือกภาพเอง =====
PICK = {
    "battery":    "battery_752.jpg",
    "biological": "biological_569.jpg",
    "cardboard":  "cardboard_968.jpg",
    "clothes":    "clothes_web_1096.jpg",
    "glass":      "c0baa6bd-e4e5-4c5e-a33b-944171c55697.jpg",                    # เว้นว่าง = เลือกไฟล์แรกอัตโนมัติ
    "metal":      "LINE_ALBUM_Dataset2_250816_66.jpg",
    "paper":      "paper_2074.jpg",
    "plastic":    "hard_20260724_094648_plastic_f0c6bac0_2.jpg",
    "shoes":      "shoes_1082.jpg",
    "trash":      "trash_69.jpg",
}
# =====================================


def pick_file(cls):
    want = PICK.get(cls, "")
    if want:
        if os.path.isabs(want) and os.path.exists(want):
            return want                                   # path เต็ม (ใช้รูปตัวเองได้)
        p = os.path.join(ROOT, cls, want)
        if os.path.exists(p):
            return p
        print(f"  [!] ไม่พบไฟล์ {want} ของประเภท {cls} — ใช้ไฟล์แรกแทน")
    d = os.path.join(ROOT, cls)
    files = sorted(f for f in os.listdir(d) if not f.startswith("hard_"))
    return os.path.join(d, files[0])


def square(path, size=420):
    """ครอบตัดเป็นจัตุรัสให้ทุกภาพขนาดเท่ากัน"""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return im.resize((size, size), Image.LANCZOS)


ncol, nrow = 5, 2
fig, axes = plt.subplots(nrow, ncol, figsize=(13.0, 5.9), dpi=200)
fig.patch.set_facecolor("white")

for i, cls in enumerate(CLASSES):
    ax = axes[i // ncol][i % ncol]
    f = pick_file(cls)
    ax.imshow(square(f))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE); sp.set_linewidth(1.8)
    ax.set_title(cls, fontproperties=TH_B, fontsize=11.5, color=TXT, pad=6)
    print(f"  {cls:11s} <- {os.path.basename(f)}")

plt.tight_layout(pad=0.7, w_pad=0.9, h_pad=1.4)
plt.savefig(OUTFILE, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.2)
print(f"\nsaved: {OUTFILE}")
