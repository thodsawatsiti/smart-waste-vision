"""
compare_processing.py
เปรียบเทียบ 4 แบบ:
1. ต้นฉบับ (Original)
2. CLAHE อย่างเดียว
3. GrabCut อย่างเดียว
4. CLAHE + GrabCut

วิธีใช้:
    python compare_processing.py <image_path>
ตัวอย่าง:
    python compare_processing.py test_photo.jpg
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize_image(img, size=640):
    """Resize ให้ด้านยาวสุดเป็น 640px"""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def apply_clahe(img):
    """ปรับ contrast แบบ adaptive"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_grabcut(img):
    """ลบ background ใส่พื้นขาว"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    white_bg = np.ones_like(img) * 255
    result = np.where(fg_mask[:, :, np.newaxis] == 1, img, white_bg)
    return result.astype(np.uint8)


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_processing.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        sys.exit(1)

    print(f"Processing: {image_path}")

    # โหลด + resize
    img = cv2.imread(image_path)
    img = resize_image(img)
    print(f"Size: {img.shape[:2]}")

    # 4 versions
    print("[1/4] Original...")
    original = img.copy()

    print("[2/4] CLAHE only...")
    clahe_only = apply_clahe(img)

    print("[3/4] GrabCut only...")
    grabcut_only = apply_grabcut(img)

    print("[4/4] CLAHE + GrabCut...")
    combined = apply_grabcut(apply_clahe(img))

    # Plot 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Image Processing Comparison", fontsize=16, fontweight='bold')

    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original (no processing)", fontsize=13)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(clahe_only, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("2. CLAHE only (contrast enhanced)", fontsize=13)
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(grabcut_only, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("3. GrabCut only (background removed)", fontsize=13)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("4. CLAHE + GrabCut (both)", fontsize=13)
    axes[1, 1].axis('off')

    plt.tight_layout()

    save_path = image_path.rsplit('.', 1)[0] + '_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.show()
    print("Done!")


if __name__ == "__main__":
    main()
