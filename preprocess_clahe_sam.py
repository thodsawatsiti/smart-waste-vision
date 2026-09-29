"""
preprocess_clahe_sam.py
สร้าง dataset ที่ผ่าน CLAHE + SAM จาก garbage_dataset_cls (raw)
-> บันทึกที่ garbage_dataset_cls_clahe_sam/  (train + val)

รองรับการรันซ้ำ: ถ้าไฟล์ปลายทางมีแล้วจะข้าม (resume ได้ถ้า crash)
รัน:
    python preprocess_clahe_sam.py
"""
import os, sys, io, glob
import cv2, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = "garbage_dataset_cls"
DST = "garbage_dataset_cls_clahe_sam"
SIZE = 640                 # ย่อด้านยาวสุดเป็น 640 (ตรงกับ imgsz ตอนเทรน)
SAM_IMGSZ = 640            # ความละเอียด mask ของ SAM


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

    classes = sorted(d for d in os.listdir(os.path.join(SRC, "train"))
                     if os.path.isdir(os.path.join(SRC, "train", d)))
    total = done = skip = 0
    for split in ["train", "val"]:
        for cls in classes:
            src = os.path.join(SRC, split, cls)
            dst = os.path.join(DST, split, cls); os.makedirs(dst, exist_ok=True)
            files = glob.glob(os.path.join(src, "*.jpg")) + glob.glob(os.path.join(src, "*.png"))
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
    print(f"\nขั้นต่อไป: python train_clahe_sam.py")


if __name__ == "__main__":
    main()
