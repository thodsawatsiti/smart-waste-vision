"""
preprocess_material_patches.py
สร้างเดตาเซตที่ครอปเฉพาะ "เนื้อพื้นผิว" ใกล้กลางภาพแบบสุ่ม (ซูมเข้า ไม่เอาทั้งรูปทรงวัตถุ)
แนวคิดจากงาน Material Recognition (MINC) ที่เทรนจาก patch แทนที่จะเป็นทั้งภาพ
เพื่อบังคับให้โมเดลโฟกัส texture/การสะท้อนแสง แทนรูปทรงโดยรวม

- train: ครอปสุ่ม 45-70% ของด้านสั้น เอนเอียงเข้ากลางภาพ (เพราะวัตถุมักอยู่กลางเฟรม)
- val: ใช้รูปเต็มเหมือนเดิม (ตอนใช้งานจริงผู้ใช้ส่งรูปเต็มมา ไม่มีใครครอปให้ก่อน)

วิธีใช้:
    python preprocess_material_patches.py
"""
import os, sys, random
import cv2

sys.stdout.reconfigure(encoding="utf-8")

SOURCE = "garbage_dataset_cls"
TARGET = "garbage_dataset_cls_patches"
MIN_RATIO, MAX_RATIO = 0.45, 0.70
SEED = 42

random.seed(SEED)


def random_patch_crop(img):
    h, w = img.shape[:2]
    side = min(h, w)
    crop_size = int(side * random.uniform(MIN_RATIO, MAX_RATIO))
    crop_size = max(crop_size, 32)

    # จุดศูนย์กลางครอป เอนเอียงเข้ากลางภาพ (สุ่มขยับได้นิดหน่อย ไม่ใช่กลางเป๊ะทุกรูป)
    cx = w // 2 + random.randint(-w // 10, w // 10)
    cy = h // 2 + random.randint(-h // 10, h // 10)

    x0 = max(0, min(w - crop_size, cx - crop_size // 2))
    y0 = max(0, min(h - crop_size, cy - crop_size // 2))
    return img[y0:y0 + crop_size, x0:x0 + crop_size]


def main():
    classes = sorted(d for d in os.listdir(os.path.join(SOURCE, "train"))
                      if os.path.isdir(os.path.join(SOURCE, "train", d)))

    total = done = skip = 0
    for split in ["train", "val"]:
        for cls in classes:
            src_dir = os.path.join(SOURCE, split, cls)
            dst_dir = os.path.join(TARGET, split, cls)
            os.makedirs(dst_dir, exist_ok=True)

            files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            for fname in files:
                total += 1
                outp = os.path.join(dst_dir, os.path.splitext(fname)[0] + ".jpg")
                if os.path.exists(outp):
                    skip += 1
                    continue

                img = cv2.imread(os.path.join(src_dir, fname))
                if img is None:
                    continue

                if split == "train":
                    out = random_patch_crop(img)
                else:
                    out = img  # val: รูปเต็ม ไม่ครอป ให้ตรงกับตอนใช้งานจริง

                cv2.imwrite(outp, out)
                done += 1
                if done % 500 == 0:
                    print(f"  ...{done} รูป (ข้าม {skip})", flush=True)

        print(f"[{split}] เสร็จ", flush=True)

    n_tr = sum(len(os.listdir(os.path.join(TARGET, "train", c))) for c in classes)
    n_va = sum(len(os.listdir(os.path.join(TARGET, "val", c))) for c in classes)
    print(f"\nเสร็จ! processed ใหม่ {done} | ข้าม {skip} | รวม {total}")
    print(f"  {TARGET}/train = {n_tr} รูป (ครอป) | val = {n_va} รูป (เต็ม ไม่ครอป)")
    print(f"\nขั้นต่อไป: python train_material_patches.py")


if __name__ == "__main__":
    main()
