"""
sgd_vs_adamw_experiment.py
การทดลอง: เปรียบเทียบ optimizer SGD vs AdamW + จูน learning rate (grid search)
บน dataset ของเราเอง (garbage_dataset_cls, raw)

คุมทุกอย่างให้เท่ากัน เปลี่ยนแค่ optimizer + lr เพื่อความเป็นธรรม (fair comparison)
- imgsz=224, epochs=25, batch=32, seed=42, cos_lr=True (เหมือนกันทุก run)
"""
import os, sys, csv, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from ultralytics import YOLO

DATA = "garbage_dataset_cls"          # raw dataset (no processing)
PROJECT = "runs/classify/sgd_experiment"
OUT = "results/sgd_experiment_summary.csv"

CONFIGS = [
    ("AdamW_lr0.001", "AdamW", 0.001),
    ("SGD_lr0.01",    "SGD",   0.01),
    ("SGD_lr0.001",   "SGD",   0.001),
    ("SGD_lr0.0001",  "SGD",   0.0001),
]

COMMON = dict(data=DATA, epochs=25, imgsz=224, batch=32, device=0,
              project=PROJECT, exist_ok=True, pretrained=True,
              cos_lr=True, seed=42, verbose=False, plots=False, val=True,
              workers=0)   # workers=0 เลี่ยงปัญหา multiprocessing บน Windows


def best_metrics(save_dir):
    csv_path = os.path.join(str(save_dir), "results.csv")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    best = max(rows, key=lambda r: float(r["metrics/accuracy_top1"]))
    return float(best["metrics/accuracy_top1"]), float(best["metrics/accuracy_top5"]), int(best["epoch"])


def main():
    os.makedirs("results", exist_ok=True)
    summary = []
    for name, opt, lr in CONFIGS:
        print(f"\n{'='*60}\n  Training: {name}  (optimizer={opt}, lr0={lr})\n{'='*60}", flush=True)
        model = YOLO("yolo11n-cls.pt")
        model.train(name=name, optimizer=opt, lr0=lr, **COMMON)
        top1, top5, ep = best_metrics(model.trainer.save_dir)   # อ่านจาก path จริงที่ ultralytics เซฟ
        summary.append((name, opt, lr, top1 * 100, top5 * 100, ep))
        print(f"  >>> {name}: Top-1={top1*100:.2f}%  Top-5={top5*100:.2f}%  (best epoch {ep})", flush=True)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "optimizer", "lr0", "top1_%", "top5_%", "best_epoch"])
        for row in summary:
            w.writerow([row[0], row[1], row[2], f"{row[3]:.2f}", f"{row[4]:.2f}", row[5]])

    print(f"\n\n{'='*60}\n  สรุปผลการทดลอง (fair: 224px, 25ep, seed=42)\n{'='*60}")
    print(f"{'Config':<18}{'Optimizer':<10}{'lr0':<10}{'Top-1':<10}{'Top-5':<10}")
    print("-" * 60)
    for name, opt, lr, t1, t5, ep in sorted(summary, key=lambda r: -r[3]):
        print(f"{name:<18}{opt:<10}{lr:<10}{t1:<10.2f}{t5:<10.2f}")
    best = max(summary, key=lambda r: r[3])
    print(f"\n  ชนะ: {best[0]}  ({best[3]:.2f}%)")
    print(f"  Saved: {OUT}")


if __name__ == "__main__":
    main()
