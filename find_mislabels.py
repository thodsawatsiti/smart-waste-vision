"""
find_mislabels.py
เดาหารูปที่อาจเลเบลผิดใน training set
วิธี: เอาโมเดลที่เทรนเสร็จแล้ว (แม่น 94%+ บน val) มา predict รูปใน train เอง
      ถ้าโมเดลมั่นใจสูง (conf > threshold) ว่าเป็นคนละ class กับโฟลเดอร์ที่มันอยู่
      แปลว่ารูปนั้นน่าสงสัยว่าเลเบลผิด (หรือเป็นเคสยากจริงๆ ก็ได้ ต้องดูรูปประกอบ)
"""
import os, sys, io, glob, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATASET = "garbage_dataset_cls"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "runs/classify/runs/garbage_cls_no_processing_v2/weights/best.pt"
CONF_THRESHOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.85
OUT_CSV = "mislabel_candidates.csv"

def imgs(split, cls):
    d = os.path.join(DATASET, split, cls)
    return glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpeg"))

def main():
    classes = sorted(d for d in os.listdir(os.path.join(DATASET, "train"))
                     if os.path.isdir(os.path.join(DATASET, "train", d)))

    from ultralytics import YOLO
    model = YOLO(MODEL)

    flags = []  # (conf, true_cls, pred_cls, path)
    pair_counts = {}

    for cls in classes:
        paths = imgs("train", cls)
        print(f"scanning train/{cls} ({len(paths)} images)...", flush=True)
        for p, res in zip(paths, model.predict(paths, stream=True, device=0, verbose=False)):
            pred_idx = int(res.probs.top1)
            pred_conf = float(res.probs.top1conf)
            pred_cls = model.names[pred_idx]
            if pred_cls != cls and pred_conf > CONF_THRESHOLD:
                flags.append((pred_conf, cls, pred_cls, p))
                key = (cls, pred_cls)
                pair_counts[key] = pair_counts.get(key, 0) + 1

    flags.sort(reverse=True)

    print("\n" + "=" * 70)
    print(f"  พบ {len(flags)} รูปที่โมเดลมั่นใจ >{CONF_THRESHOLD:.0%} ว่าเลเบลในโฟลเดอร์ผิด")
    print("=" * 70)

    print("\nสรุปคู่ที่สับสนบ่อย (เลเบลเดิม -> โมเดลว่าเป็น) : จำนวน")
    for (true_c, pred_c), cnt in sorted(pair_counts.items(), key=lambda x: -x[1]):
        print(f"  {true_c:12s} -> {pred_c:12s} : {cnt}")

    print(f"\nTop 20 ที่มั่นใจสุด (ควรเปิดดูรูปจริงเพื่อยืนยัน):")
    for conf, true_c, pred_c, p in flags[:20]:
        print(f"  conf={conf:.3f}  {true_c} -> {pred_c}   {p}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["confidence", "labeled_as", "model_thinks", "path"])
        for conf, true_c, pred_c, p in flags:
            w.writerow([f"{conf:.4f}", true_c, pred_c, p])
    print(f"\nบันทึกรายการทั้งหมดลง {OUT_CSV} แล้ว ({len(flags)} แถว)")

if __name__ == "__main__":
    main()
