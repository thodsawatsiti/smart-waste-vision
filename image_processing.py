"""
image_processing.py
====================
Custom Image Processing Pipeline สำหรับ Garbage Classification
ทำเองทุกขั้นตอนด้วย OpenCV (ไม่ใช้ rembg)

แสดง visualization ทุก step เพื่อให้เห็นกระบวนการ

Usage:
    python image_processing.py <image_path>
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def load_image(path):
    """Step 0: โหลดรูป"""
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb


def resize_image(img, size=640):
    """Step 1: Resize ให้เป็นขนาดมาตรฐาน"""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def color_analysis(img_rgb):
    """Step 2: วิเคราะห์สี - Color Histogram"""
    colors = ('r', 'g', 'b')
    hist_data = {}
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        hist_data[color] = hist.flatten()
    return hist_data


def grayscale_and_blur(img):
    """Step 3: แปลงเป็น Grayscale + Gaussian Blur"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blurred


def apply_clahe(gray):
    """Step 4: CLAHE - Contrast Limited Adaptive Histogram Equalization"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def edge_detection(blurred):
    """Step 5: Edge Detection - Canny"""
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def thresholding(gray):
    """Step 6: Thresholding - Otsu's Method"""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def morphological_operations(binary):
    """Step 7: Morphological Operations - ปรับ mask ให้สมบูรณ์"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # ปิดรู (Close)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    # เปิดลบ noise (Open)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
    return closed, opened


def grabcut_segmentation(img):
    """Step 8: GrabCut - Background Removal แบบ semi-automatic"""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # กำหนด bounding box (ตรงกลาง 80% ของรูป)
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # สร้าง mask: 0,2 = background, 1,3 = foreground
    fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = img * fg_mask[:, :, np.newaxis]

    return fg_mask, result


def contour_analysis(binary, img_rgb):
    """Step 9: Contour Detection - หา contour ของวัตถุ"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # เรียงจากใหญ่ไปเล็ก
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    img_contour = img_rgb.copy()
    cv2.drawContours(img_contour, contours[:5], -1, (0, 255, 0), 2)

    # หา bounding box ของ contour ใหญ่สุด
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.rectangle(img_contour, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return contours, img_contour


def create_segmentation_map(img, fg_mask, k=5):
    """Step 10: Segmentation Map - ใช้ K-Means Clustering แยกส่วนของภาพ"""
    h, w = img.shape[:2]

    # แปลงเป็น RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # เตรียมข้อมูลสำหรับ K-Means (เฉพาะ foreground)
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)

    # K-Means Clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # สร้าง segmentation map
    seg_map = labels.reshape(h, w).astype(np.uint8)

    # Background = class สุดท้าย
    seg_map[fg_mask == 0] = k

    # สร้างรูป segmented โดยใช้สีสดแยกแต่ละ cluster ให้เห็นชัด
    CLUSTER_COLORS = [
        [255, 80,  80],   # แดง
        [80,  220, 80],   # เขียว
        [80,  160, 255],  # ฟ้า
        [255, 200, 0],    # เหลือง
        [200, 80,  255],  # ม่วง
    ]
    flat_labels = labels.flatten()
    colored = np.array([CLUSTER_COLORS[l % len(CLUSTER_COLORS)] for l in flat_labels], dtype=np.uint8)
    segmented_img = colored.reshape(h, w, 3)
    segmented_img[fg_mask == 0] = [30, 30, 30]  # พื้นหลังดำ

    return seg_map, segmented_img, k


def plot_full_pipeline(image_path, save_path=None):
    """แสดง visualization ทุกขั้นตอน"""
    print(f"Processing: {image_path}")
    print("=" * 50)

    # Step 0: Load
    img, img_rgb = load_image(image_path)
    print("[0] Image loaded")

    # Step 1: Resize
    img_resized = resize_image(img)
    img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    print(f"[1] Resized: {img.shape[:2]} -> {img_resized.shape[:2]}")

    # Step 2: Color Histogram
    hist_data = color_analysis(img_resized_rgb)
    print("[2] Color histogram computed")

    # Step 3: Grayscale + Blur
    gray, blurred = grayscale_and_blur(img_resized)
    print("[3] Grayscale + Gaussian Blur")

    # Step 4: CLAHE
    enhanced = apply_clahe(gray)
    print("[4] CLAHE applied")

    # Step 5: Edge Detection
    edges = edge_detection(blurred)
    print("[5] Canny Edge Detection")

    # Step 6: Thresholding
    binary = thresholding(gray)
    print("[6] Otsu's Thresholding")

    # Step 7: Morphological Operations
    closed, opened = morphological_operations(binary)
    print("[7] Morphological Operations")

    # Step 8: GrabCut
    fg_mask, grabcut_result = grabcut_segmentation(img_resized)
    grabcut_rgb = cv2.cvtColor(grabcut_result, cv2.COLOR_BGR2RGB)
    print("[8] GrabCut Segmentation")

    # Step 9: Contour Analysis
    contours, img_contour = contour_analysis(opened, img_resized_rgb.copy())
    print(f"[9] Found {len(contours)} contours")

    # Step 10: Segmentation Map (K-Means)
    seg_map, segmented_img, k = create_segmentation_map(img_resized, fg_mask)
    print(f"[10] K-Means Segmentation (k={k})")

    # ============================
    # Plot ทั้งหมด
    # ============================
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    fig.suptitle("Image Processing Pipeline for Garbage Classification",
                 fontsize=16, fontweight='bold')

    # Row 1
    axes[0, 0].imshow(img_resized_rgb)
    axes[0, 0].set_title("1. Original (Resized)")
    axes[0, 0].axis('off')

    for color, data in hist_data.items():
        axes[0, 1].plot(data, color=color, alpha=0.7)
    axes[0, 1].set_title("2. Color Histogram")
    axes[0, 1].set_xlabel("Pixel Value")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend(['Red', 'Green', 'Blue'])

    axes[0, 2].imshow(gray, cmap='gray')
    axes[0, 2].set_title("3. Grayscale")
    axes[0, 2].axis('off')

    # Row 2
    axes[1, 0].imshow(enhanced, cmap='gray')
    axes[1, 0].set_title("4. CLAHE Enhanced")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(edges, cmap='gray')
    axes[1, 1].set_title("5. Canny Edge Detection")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(binary, cmap='gray')
    axes[1, 2].set_title("6. Otsu's Thresholding")
    axes[1, 2].axis('off')

    # Row 3
    axes[2, 0].imshow(closed, cmap='gray')
    axes[2, 0].set_title("7a. Morphological Close")
    axes[2, 0].axis('off')

    axes[2, 1].imshow(opened, cmap='gray')
    axes[2, 1].set_title("7b. Morphological Open")
    axes[2, 1].axis('off')

    axes[2, 2].imshow(img_contour)
    axes[2, 2].set_title(f"8. Contours ({len(contours)} found)")
    axes[2, 2].axis('off')

    # Row 4
    axes[3, 0].imshow(fg_mask, cmap='gray')
    axes[3, 0].set_title("9. GrabCut Mask")
    axes[3, 0].axis('off')

    axes[3, 1].imshow(grabcut_rgb)
    axes[3, 1].set_title("10. Background Removed")
    axes[3, 1].axis('off')

    # Segmentation Map (K-Means)
    axes[3, 2].imshow(segmented_img)
    axes[3, 2].set_title(f"11. K-Means Segmentation (k={k})")
    axes[3, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
    else:
        save_path = image_path.rsplit('.', 1)[0] + '_pipeline.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")

    plt.show()
    print("Done!")


def smart_preprocess(img_or_path, save_visualization=False):
    """
    Smart preprocessing สำหรับทั้ง train + predict
    Pipeline: Resize → CLAHE → GrabCut (ใช้ผลเสมอ ไม่มี fallback)

    - CLAHE: ทำทุกรูป (baseline preprocessing)
    - GrabCut: ทำเมื่อ mask ดี (10-90% ของภาพ)
    - Fallback: ถ้า GrabCut แย่ → ใช้ CLAHE-only

    Args:
        img_or_path: รูป (np.array) หรือ path
        save_visualization: บันทึกรูปเปรียบเทียบ

    Returns:
        np.array: รูปที่ preprocess แล้ว (BGR)
    """
    # โหลดรูป
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path)
        if img is None:
            return None
    else:
        img = img_or_path

    # Step 1: Resize
    img = resize_image(img, 640)
    h, w = img.shape[:2]

    # Step 2: CLAHE (baseline - ทำทุกรูป)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Step 3: GrabCut (ทำทุกรูปเสมอ — ไม่มี fallback ตามคุณภาพ mask แล้ว
    # เหตุผล: ถ้าบางรูปตัด background ได้ บางรูป fallback ไปใช้ CLAHE-only
    # (ยังมี background เดิม) เทรนนิ่งเซตจะมีสองสไตล์ปนกัน เกิด spike ของข้อมูลได้)
    try:
        mask = np.zeros((h, w), np.uint8)
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(clahe_img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # ใช้ผล GrabCut เสมอ ไม่ว่าขนาด mask จะเป็นเท่าไหร่
        white_bg = np.ones_like(clahe_img) * 255
        result = np.where(fg_mask[:, :, np.newaxis] == 1, clahe_img, white_bg)
        return result.astype(np.uint8)

    except Exception:
        # เคสนี้คือ GrabCut error จริงๆ (เช่นรูปเสีย) ไม่ใช่ fallback ตามคุณภาพ mask
        return clahe_img


def process_for_prediction(image_path):
    """
    [DEPRECATED] ใช้ smart_preprocess() แทน
    เก็บไว้เพื่อ backward compatibility
    """
    img = cv2.imread(image_path)
    img = resize_image(img, 640)

    # GrabCut
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # ใส่พื้นหลังขาว
    white_bg = np.ones_like(img) * 255
    result = np.where(fg_mask[:, :, np.newaxis] == 1, img, white_bg)

    return result.astype(np.uint8)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_processing.py <image_path>")
        print("Example: python image_processing.py test_photo.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        sys.exit(1)

    plot_full_pipeline(image_path)
