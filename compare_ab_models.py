"""
compare_ab_models.py
A/B Comparison ระหว่าง 2 models:
- Model A: garbage_cls_no_processing (ไม่ใช้ CLAHE + GrabCut)
- Model B: garbage_cls_with_processing (ใช้ CLAHE + GrabCut)

แสดง:
1. Training curves เปรียบเทียบ
2. Confusion matrix side-by-side
3. Per-class accuracy
4. สรุปข้อค้นพบ

วิธีใช้:
    python compare_ab_models.py
"""

import os
import sys
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"C:\Users\popjr\Desktop\garbage\garbage-detection-yolov8\runs\classify\runs"
MODEL_A_DIR = os.path.join(BASE_DIR, "garbage_cls_no_processing")
MODEL_B_DIR = os.path.join(BASE_DIR, "garbage_cls_with_processing")
OUTPUT_DIR = r"C:\Users\popjr\Desktop\garbage\garbage-detection-yolov8\results"

CLASSES = ['battery', 'biological', 'cardboard', 'clothes', 'glass',
           'metal', 'paper', 'plastic', 'shoes', 'trash']


def read_results(model_dir):
    """อ่าน results.csv ดู accuracy ของ best epoch"""
    csv_path = os.path.join(model_dir, "results.csv")
    if not os.path.exists(csv_path):
        return None

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    # หา best epoch (top1 accuracy สูงสุด)
    best_row = max(rows, key=lambda r: float(r['metrics/accuracy_top1']))

    return {
        'total_epochs': len(rows),
        'best_epoch': int(best_row['epoch']),
        'top1': float(best_row['metrics/accuracy_top1']),
        'top5': float(best_row['metrics/accuracy_top5']),
        'train_loss': float(best_row['train/loss']),
        'val_loss': float(best_row['val/loss']),
        'rows': rows,
    }


def plot_training_curves(model_a, model_b, output_path):
    """กราฟ training curves เปรียบเทียบ"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Curves: With vs Without Image Processing',
                 fontsize=15, fontweight='bold')

    # Top-1 Accuracy
    epochs_a = [int(r['epoch']) for r in model_a['rows']]
    acc_a = [float(r['metrics/accuracy_top1']) * 100 for r in model_a['rows']]
    epochs_b = [int(r['epoch']) for r in model_b['rows']]
    acc_b = [float(r['metrics/accuracy_top1']) * 100 for r in model_b['rows']]

    axes[0].plot(epochs_a, acc_a, label='WITHOUT processing (Model A)',
                 color='#10b981', linewidth=2)
    axes[0].plot(epochs_b, acc_b, label='WITH CLAHE+GrabCut (Model B)',
                 color='#ef4444', linewidth=2)
    axes[0].set_title('Top-1 Accuracy', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)

    # Val Loss
    loss_a = [float(r['val/loss']) for r in model_a['rows']]
    loss_b = [float(r['val/loss']) for r in model_b['rows']]

    axes[1].plot(epochs_a, loss_a, label='WITHOUT processing (Model A)',
                 color='#10b981', linewidth=2)
    axes[1].plot(epochs_b, loss_b, label='WITH CLAHE+GrabCut (Model B)',
                 color='#ef4444', linewidth=2)
    axes[1].set_title('Validation Loss', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_confusion_comparison(output_path):
    """แสดง Confusion Matrix ของทั้ง 2 models side-by-side"""
    cm_a = os.path.join(MODEL_A_DIR, "confusion_matrix.png")
    cm_b = os.path.join(MODEL_B_DIR, "confusion_matrix.png")

    if not os.path.exists(cm_a) or not os.path.exists(cm_b):
        print("ไม่พบ confusion_matrix.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    img_a = mpimg.imread(cm_a)
    img_b = mpimg.imread(cm_b)

    axes[0].imshow(img_a)
    axes[0].set_title('Model A: WITHOUT Image Processing',
                      fontsize=14, fontweight='bold', color='#10b981')
    axes[0].axis('off')

    axes[1].imshow(img_b)
    axes[1].set_title('Model B: WITH CLAHE + GrabCut',
                      fontsize=14, fontweight='bold', color='#ef4444')
    axes[1].axis('off')

    plt.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary(model_a, model_b):
    """พิมพ์สรุปเปรียบเทียบ"""
    print("\n" + "=" * 70)
    print("  A/B Comparison Summary")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Model A (Raw)':<20} {'Model B (Processed)':<20}")
    print("-" * 70)
    print(f"{'Top-1 Accuracy':<25} {model_a['top1']*100:>6.2f}%             {model_b['top1']*100:>6.2f}%")
    print(f"{'Top-5 Accuracy':<25} {model_a['top5']*100:>6.2f}%             {model_b['top5']*100:>6.2f}%")
    print(f"{'Train Loss':<25} {model_a['train_loss']:>6.4f}              {model_b['train_loss']:>6.4f}")
    print(f"{'Val Loss':<25} {model_a['val_loss']:>6.4f}              {model_b['val_loss']:>6.4f}")
    print(f"{'Best Epoch':<25} {model_a['best_epoch']:>6}                {model_b['best_epoch']:>6}")
    print(f"{'Total Epochs':<25} {model_a['total_epochs']:>6}                {model_b['total_epochs']:>6}")

    # ใครชนะ?
    print("\n" + "=" * 70)
    diff_top1 = (model_a['top1'] - model_b['top1']) * 100
    if diff_top1 > 0:
        winner = "Model A (WITHOUT processing)"
        print(f"  🏆 ผู้ชนะ: {winner}")
        print(f"     ดีกว่า {abs(diff_top1):.2f}% on Top-1 Accuracy")
    elif diff_top1 < 0:
        winner = "Model B (WITH CLAHE + GrabCut)"
        print(f"  🏆 ผู้ชนะ: {winner}")
        print(f"     ดีกว่า {abs(diff_top1):.2f}% on Top-1 Accuracy")
    else:
        print(f"  🤝 เสมอ")
    print("=" * 70)


def main():
    print("=" * 70)
    print("  A/B Model Comparison")
    print("=" * 70)

    # อ่านผลทั้ง 2 models
    print(f"\n[1] Reading Model A: {MODEL_A_DIR}")
    model_a = read_results(MODEL_A_DIR)
    print(f"[2] Reading Model B: {MODEL_B_DIR}")
    model_b = read_results(MODEL_B_DIR)

    if model_a is None or model_b is None:
        print("ไม่พบ results.csv ของอย่างน้อย 1 model")
        return

    # สรุปตัวเลข
    print_summary(model_a, model_b)

    # สร้างกราฟ
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n[3] กำลังสร้างกราฟเปรียบเทียบ...")

    plot_training_curves(model_a, model_b,
                          os.path.join(OUTPUT_DIR, "ab_training_curves.png"))
    plot_confusion_comparison(os.path.join(OUTPUT_DIR, "ab_confusion_matrix.png"))

    print(f"\n✅ เสร็จเรียบร้อย!")
    print(f"   ดูผลที่: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
