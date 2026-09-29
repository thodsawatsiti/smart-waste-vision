"""
sweep_finetune_epochs.py
ทดสอบเทียบว่า fine-tune กี่ epoch ถึงจะดีที่สุด (แทนที่จะเดา 15 ลอยๆ)
เริ่มจาก base checkpoint เดียวกันทุกครั้ง (ไม่ต่อยอดกันเอง) แล้ววัดทั้ง 2 อย่าง:
  - val set หลัก (เช็คว่า catastrophic forgetting ไหม)
  - 16 รูป holdout จริง (เช็คว่าแก้ pose gap ได้จริงไหม)

รัน:
    python sweep_finetune_epochs.py
"""
import os, sys, glob, io, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from ultralytics import YOLO

BASE_MODEL = "runs/classify/runs/garbage_cls_no_processing_v2/weights/best.pt"
DATASET = "garbage_dataset_cls"
HARD_VAL = "hard_val"
EPOCH_VALUES = [5, 10, 15, 20, 25]
RESULTS_CSV = "sweep_finetune_epochs_results.csv"


def eval_holdout(model_path):
    model = YOLO(model_path)
    classes = sorted(d for d in os.listdir(HARD_VAL) if os.path.isdir(os.path.join(HARD_VAL, d)))
    total, correct = 0, 0
    for cls in classes:
        for p in glob.glob(os.path.join(HARD_VAL, cls, "*")):
            res = model.predict(p, verbose=False)[0]
            pred = model.names[int(res.probs.top1)]
            total += 1
            correct += int(pred == cls)
    return correct, total


def main():
    results = []
    for ep in EPOCH_VALUES:
        run_name = f"garbage_cls_hardexamples_ep{ep}"
        print(f"\n{'='*60}\nfine-tune {ep} epoch -> {run_name}\n{'='*60}", flush=True)

        best_pt = f"runs/classify/runs/classify/runs/{run_name}/weights/best.pt"
        if os.path.exists(best_pt):
            print(f"  (ข้าม — เทรนไว้แล้วที่ {best_pt})", flush=True)
        else:
            model = YOLO(BASE_MODEL)  # โหลด base ใหม่ทุกรอบ ไม่ต่อยอดจากรอบก่อนหน้า
            model.train(
                data=DATASET,
                epochs=ep,
                lr0=0.0005,
                patience=ep,  # ไม่ให้ early stop ตัดก่อนครบ เพราะอยากรู้ผลที่ epoch นั้นจริงๆ
                device=0,
                project="runs/classify/runs",
                name=run_name,
                exist_ok=True,
                verbose=False,
            )

        val_metrics = YOLO(best_pt).val(data=DATASET, verbose=False)
        val_top1 = float(val_metrics.top1)
        hc, ht = eval_holdout(best_pt)
        holdout_acc = hc / ht * 100

        print(f"  epoch={ep}  val_top1={val_top1*100:.2f}%  holdout={hc}/{ht}={holdout_acc:.1f}%", flush=True)
        results.append((ep, val_top1 * 100, hc, ht, holdout_acc))

    print(f"\n\n{'='*70}")
    print("  สรุปผลเทียบทุก epoch")
    print(f"{'='*70}")
    print(f"  {'epoch':>6} {'val_top1':>10} {'holdout':>12} {'holdout%':>10}")
    for ep, val_top1, hc, ht, holdout_acc in results:
        print(f"  {ep:>6} {val_top1:>9.2f}% {hc:>4}/{ht:<4}    {holdout_acc:>8.1f}%")

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "val_top1_pct", "holdout_correct", "holdout_total", "holdout_pct"])
        for row in results:
            w.writerow(row)
    print(f"\nบันทึกผลเต็มที่: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
