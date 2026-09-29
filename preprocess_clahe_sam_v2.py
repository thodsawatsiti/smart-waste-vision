"""
preprocess_clahe_sam_v2.py
เหมือน preprocess_clahe_sam.py แต่:
- ไม่มี fallback ตามคุณภาพ mask แล้ว (ไม่เช็ค area 5-90% หรือต้องอยู่กลางภาพ)
  -> เลือก mask ที่ใหญ่ที่สุดที่ SAM หาเจอเสมอ (ยกเว้นไม่เจอ mask เลย ถึงจะคืนภาพ CLAHE เฉย ๆ)
- พื้นหลังเป็นสีขาว (แทนที่จะเป็นดำแบบเดิม) ให้ตรงกับ GrabCut v2
- ใช้ garbage_dataset_cls ปัจจุบัน (รวม hard examples จาก feedback loop แล้ว) เหมือน GrabCut v2

วิธีใช้:
    python preprocess_clahe_sam_v2.py
"""
import os, sys, io, glob
import cv2, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = "garbage_dataset_cls"
DST = "garbage_dataset_cls_clahe_sam_v2"
SIZE = 640
SAM_IMGSZ = 640


def clahe_color(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def main():
    from ultralytics import FastSAM
    sam = FastSAM("FastSAM-s.pt")

    def sam_removebg(img):
        h, w = img.shape[:2]
        try:
            res = sam(img, device=0, retina_masks=True, imgsz=SAM_IMGSZ,
                      conf=0.4, iou=0.9, verbose=False)
            if res[0].masks is None:
                return img  # ไม่เจอ mask เลย ไม่ใช่ fallback เชิงคุณภาพ แต่ไม่มีอะไรให้ใช้จริงๆ
            masks = res[0].masks.data.cpu().numpy()
            best, ba = None, 0
            for m in masks:
                m = cv2.resize(m.astype("uint8"), (w, h), interpolation=cv2.INTER_NEAREST)
                if m.sum() > ba:
                    best, ba = m, m.sum()
            if best is None:
                return img
            white_bg = np.ones_like(img) * 255
            return np.where(best[:, :, None] == 1, img, white_bg).astype(np.uint8)
        except Exception:
            return img

    classes = sorted(d for d in os.listdir(os.path.join(SRC, "train"))
                     if os.path.isdir(os.path.join(SRC, "train", d)))
    total = done = skip = 0
    for split in ["train", "val"]:
        for cls in classes:
            src = os.path.join(SRC, split, cls)
            dst = os.path.join(DST, split, cls); os.makedirs(dst, exist_ok=True)
            files = glob.glob(os.path.join(src, "*.jpg")) + glob.glob(os.path.join(src, "*.png")) + glob.glob(os.path.join(src, "*.jpeg"))
            for f in files:
                total += 1
                outp = os.path.join(dst, os.path.splitext(os.path.basename(f))[0] + ".jpg")
                if os.path.exists(outp):
                    skip += 1; continue
                img = cv2.imread(f)
                if img is None:
                    continue
                h, w = img.shape[:2]; s = SIZE / max(h, w)
                if s < 1:
                    img = cv2.resize(img, (int(w*s), int(h*s)))
                out = sam_removebg(clahe_color(img))
                cv2.imwrite(outp, out)
                done += 1
                if done % 200 == 0:
                    print(f"  ...{done} รูป (ข้าม {skip})", flush=True)
        print(f"[{split}] เสร็จ", flush=True)

    n_tr = sum(len(os.listdir(os.path.join(DST, "train", c))) for c in classes)
    n_va = sum(len(os.listdir(os.path.join(DST, "val", c))) for c in classes)
    print(f"\nเสร็จ! processed ใหม่ {done} | ข้าม {skip} | รวม {total}")
    print(f"  {DST}/train = {n_tr} รูป | val = {n_va} รูป")
    print(f"\nขั้นต่อไป: python train_clahe_sam_v2.py")


if __name__ == "__main__":
    main()
