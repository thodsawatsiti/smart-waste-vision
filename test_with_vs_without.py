"""
test_with_vs_without.py
ทดสอบรูปเดียวกัน เปรียบเทียบผลทำนาย:
- Model A (No Processing): predict รูปต้นฉบับ
- Model B (With Processing): apply CLAHE + GrabCut ก่อน predict

แสดง:
1. รูปต้นฉบับ
2. รูปหลัง CLAHE + GrabCut
3. ผลทำนาย Top-5 ของทั้ง 2 models

วิธีใช้:
    python test_with_vs_without.py <image_path>
    python test_with_vs_without.py <image_path1> <image_path2> ...   # หลายรูป
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

sys.stdout.reconfigure(encoding='utf-8')

# Models
MODEL_A_PATH = r"C:\Users\popjr\Desktop\garbage\garbage-detection-yolov8\runs\classify\runs\garbage_cls_no_processing\weights\best.pt"
MODEL_B_PATH = r"C:\Users\popjr\Desktop\garbage\garbage-detection-yolov8\runs\classify\runs\garbage_cls_with_processing\weights\best.pt"

OUTPUT_DIR = r"C:\Users\popjr\Desktop\garbage\garbage-detection-yolov8\results"


def predict(model, img):
    """ทำนาย + return top-5"""
    results = model.predict(img, verbose=False)
    probs = results[0].probs
    if probs is None:
        return None

    top5_idx = probs.top5
    top5_conf = probs.top5conf.tolist()
    return [(model.names[int(idx)], conf * 100) for idx, conf in zip(top5_idx, top5_conf)]


def make_comparison_figure(image_path, model_a, model_b, save_path):
    """สร้างรูปเปรียบเทียบ"""
    from image_processing import smart_preprocess

    # Load + resize image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    scale = 640 / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Apply processing
    processed = smart_preprocess(img)
    if processed is None:
        processed = img

    # Predict
    pred_a = predict(model_a, img)         # Model A: predict on RAW
    pred_b = predict(model_b, processed)   # Model B: predict on PROCESSED

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Comparison: {os.path.basename(image_path)}',
                 fontsize=15, fontweight='bold')

    # === Top-left: Original image ===
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('1. Original Image\n(input to Model A)',
                          fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # === Top-right: Processed image ===
    axes[0, 1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('2. After CLAHE + GrabCut\n(input to Model B)',
                          fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # === Bottom-left: Model A predictions ===
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Model A: WITHOUT Processing',
                          fontsize=13, fontweight='bold', color='#10b981')

    if pred_a:
        # Bar chart แทน text
        names = [p[0] for p in pred_a]
        confs = [p[1] for p in pred_a]
        colors = ['#10b981' if i == 0 else '#86efac' for i in range(len(names))]
        y_pos = np.arange(len(names))
        axes[1, 0].barh(y_pos, confs, color=colors, edgecolor='white')
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(names, fontsize=11)
        axes[1, 0].invert_yaxis()
        axes[1, 0].set_xlim(0, 100)
        axes[1, 0].set_xlabel('Confidence (%)', fontsize=10)
        for i, (name, conf) in enumerate(pred_a):
            axes[1, 0].text(conf + 1, i, f'{conf:.1f}%',
                            va='center', fontsize=10, fontweight='bold')
        axes[1, 0].axis('on')

    # === Bottom-right: Model B predictions ===
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Model B: WITH CLAHE + GrabCut',
                          fontsize=13, fontweight='bold', color='#ef4444')

    if pred_b:
        names = [p[0] for p in pred_b]
        confs = [p[1] for p in pred_b]
        colors = ['#ef4444' if i == 0 else '#fca5a5' for i in range(len(names))]
        y_pos = np.arange(len(names))
        axes[1, 1].barh(y_pos, confs, color=colors, edgecolor='white')
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(names, fontsize=11)
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_xlim(0, 100)
        axes[1, 1].set_xlabel('Confidence (%)', fontsize=10)
        for i, (name, conf) in enumerate(pred_b):
            axes[1, 1].text(conf + 1, i, f'{conf:.1f}%',
                            va='center', fontsize=10, fontweight='bold')
        axes[1, 1].axis('on')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

    # Print summary
    print(f"\n  Model A (No Processing): {pred_a[0][0]} ({pred_a[0][1]:.1f}%)")
    print(f"  Model B (With Processing): {pred_b[0][0]} ({pred_b[0][1]:.1f}%)")
    if pred_a[0][0] == pred_b[0][0]:
        print(f"  → ผลทำนายเหมือนกัน")
    else:
        print(f"  → ผลทำนายต่างกัน!")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_with_vs_without.py <image_path> [more images...]")
        sys.exit(1)

    image_paths = sys.argv[1:]

    # Check models exist
    if not os.path.exists(MODEL_A_PATH):
        print(f"ไม่พบ Model A: {MODEL_A_PATH}")
        sys.exit(1)
    if not os.path.exists(MODEL_B_PATH):
        print(f"ไม่พบ Model B: {MODEL_B_PATH}")
        sys.exit(1)

    # Load models
    print("Loading models...")
    model_a = YOLO(MODEL_A_PATH)
    model_b = YOLO(MODEL_B_PATH)
    print("Models loaded!")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each image
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"\n❌ ไม่พบไฟล์: {image_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Testing: {os.path.basename(image_path)}")
        print('=' * 60)

        # Output filename
        basename = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(OUTPUT_DIR, f"compare_{basename}.png")

        make_comparison_figure(image_path, model_a, model_b, save_path)

    print(f"\n{'=' * 60}")
    print(f"✅ เสร็จเรียบร้อย — ดูผลที่: {OUTPUT_DIR}")
    print('=' * 60)


if __name__ == "__main__":
    main()
