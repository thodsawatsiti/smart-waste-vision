"""
test_external_model.py
เปรียบเทียบผลทำนายระหว่างโมเดลปัจจุบัน (classify, เทรนจากรูปจริงที่ user อัพโหลด)
กับโมเดลสำเร็จรูปจาก Hugging Face (kendrickfff/waste-classification-yolov8-ken, detect, 12 classes)
บนรูปขยะไทยจริงที่เก็บไว้ใน uploads/
"""
import glob
import os
from ultralytics import YOLO

CURRENT_MODEL = "runs/classify/runs/garbage_cls_no_processing_v2/weights/best.pt"
UPLOAD_DIR = "uploads"

# โมเดลนอกที่จะทดสอบ: (path, mapping ชื่อคลาสของโมเดลนอก -> คลาสของโปรเจกต์นี้)
EXTERNAL_MODELS = {
    "yolov8n-waste-12cls (HF kendrickfff)": (
        "external_models/yolov8n-waste-12cls-best.pt",
        {"brown-glass": "glass", "green-glass": "glass", "white-glass": "glass"},
    ),
    "trash-detection-yolo11n (HF Alope)": (
        "external_models/trash-detection-yolo11n-best.pt",
        {},  # glass, paper, plastic, trash ตรงชื่ออยู่แล้ว
    ),
    "recycle-yolo11 (HF huggingleg12, เกาหลี)": (
        "external_models/recycle-yolo11-best.pt",
        {
            "paper_pack": "paper", "paper_cup": "paper",
            "can": "metal",
            "reusable_glass": "glass", "colored_glass": "glass",
            "pet": "plastic", "vinyl": "plastic", "styrofoam": "trash",
        },
    ),
}


def make_normalizer(mapping):
    return lambda cls_name: mapping.get(cls_name, cls_name)


def main():
    paths = sorted(
        glob.glob(os.path.join(UPLOAD_DIR, "*.jpg"))
        + glob.glob(os.path.join(UPLOAD_DIR, "*.jpeg"))
        + glob.glob(os.path.join(UPLOAD_DIR, "*.png"))
    )
    if not paths:
        print("ไม่พบรูปใน uploads/")
        return

    print(f"พบ {len(paths)} รูป กำลังโหลดโมเดล...")
    current = YOLO(CURRENT_MODEL)

    results_summary = []

    for label, (model_path, mapping) in EXTERNAL_MODELS.items():
        normalize = make_normalizer(mapping)
        external = YOLO(model_path)

        agree = 0
        disagree = 0
        external_no_detect = 0

        print(f"\n\n########## {label} ##########")
        print(f"{'ไฟล์':45s} {'โมเดลปัจจุบัน':22s} {'โมเดลนอก':22s} ตรงกัน?")
        print("-" * 110)

        for p in paths:
            fname = os.path.basename(p)

            r1 = current.predict(p, verbose=False)[0]
            cur_cls = current.names[int(r1.probs.top1)]
            cur_conf = float(r1.probs.top1conf)

            r2 = external.predict(p, verbose=False, conf=0.1)[0]
            if len(r2.boxes) == 0:
                ext_cls = "(ไม่เจอวัตถุ)"
                ext_conf = 0.0
                external_no_detect += 1
                match = "-"
            else:
                best_box = max(r2.boxes, key=lambda b: float(b.conf))
                ext_cls_raw = external.names[int(best_box.cls)]
                ext_cls = normalize(ext_cls_raw)
                ext_conf = float(best_box.conf)
                if normalize(cur_cls) == ext_cls:
                    agree += 1
                    match = "✓"
                else:
                    disagree += 1
                    match = "✗"

            print(f"{fname:45s} {cur_cls + f' ({cur_conf:.0%})':22s} {ext_cls + f' ({ext_conf:.0%})':22s} {match}")

        total_compared = agree + disagree
        print("-" * 110)
        rate = agree / total_compared if total_compared else 0
        print(f"สรุป: เทียบได้ {total_compared}/{len(paths)} รูป (อีก {external_no_detect} รูปไม่เจอวัตถุ) | ตรงกัน {agree}/{total_compared} ({rate:.1%})")
        results_summary.append((label, agree, disagree, external_no_detect, rate))

    print("\n\n" + "=" * 60)
    print("สรุปรวมทุกโมเดล")
    print("=" * 60)
    for label, agree, disagree, no_detect, rate in results_summary:
        print(f"{label:45s} ตรงกัน {rate:.1%}  (agree={agree}, disagree={disagree}, no_detect={no_detect})")


if __name__ == "__main__":
    main()
