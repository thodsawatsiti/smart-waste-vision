"""
add_images.py
เพิ่มรูปใหม่เข้า dataset โดยให้ priority รูปใหม่
ถ้าเกิน 1000 จะลบรูปเก่าออกอัตโนมัติ

วิธีใช้:
    python add_images.py <class_name> <folder_path>

ตัวอย่าง:
    python add_images.py trash "C:/Users/popjr/Desktop/new garbage/trash"
    python add_images.py plastic "C:/Users/popjr/Desktop/new images/plastic"
"""

import os
import sys
import shutil
import random

DATASET = r"C:\Users\popjr\Desktop\garbage\garbage-detection-yolov8\garbage-classification-v2\garbage-dataset"
MAX = 1000
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG'}


def rename_files(folder, class_name, start_num):
    """เปลี่ยนชื่อไฟล์ให้เป็น class_N.jpg"""
    files = sorted([f for f in os.listdir(folder) if os.path.splitext(f)[1] in EXTENSIONS])

    # pass 1: temp
    for i, f in enumerate(files):
        os.rename(os.path.join(folder, f), os.path.join(folder, f'__temp_{i}.jpg'))

    # pass 2: final
    temps = sorted([f for f in os.listdir(folder) if f.startswith('__temp_')])
    renamed = []
    for i, f in enumerate(temps):
        new_name = f'{class_name}_{start_num + i}.jpg'
        os.rename(os.path.join(folder, f), os.path.join(folder, new_name))
        renamed.append(new_name)

    return renamed


def main():
    if len(sys.argv) < 3:
        print("วิธีใช้: python add_images.py <class_name> <folder_path>")
        print('ตัวอย่าง: python add_images.py trash "C:/Users/popjr/Desktop/new garbage/trash"')
        sys.exit(1)

    class_name = sys.argv[1]
    src_folder = sys.argv[2]
    dst_folder = os.path.join(DATASET, class_name)

    if not os.path.exists(src_folder):
        print(f"ไม่พบ folder: {src_folder}")
        sys.exit(1)

    if not os.path.exists(dst_folder):
        print(f"ไม่พบ class '{class_name}' ใน dataset")
        sys.exit(1)

    # นับรูปเดิมใน dataset
    old_files = [f for f in os.listdir(dst_folder) if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png'}]
    new_src_files = [f for f in os.listdir(src_folder) if os.path.splitext(f)[1] in EXTENSIONS]

    print(f"\nclass: {class_name}")
    print(f"รูปเดิมใน dataset: {len(old_files)}")
    print(f"รูปใหม่ที่จะเพิ่ม: {len(new_src_files)}")

    # เปลี่ยนชื่อรูปใหม่ให้ต่อจากเลขเดิม
    start_num = len(old_files) + 1
    renamed = rename_files(src_folder, class_name, start_num)

    # copy รูปใหม่เข้า dataset
    for f in renamed:
        shutil.copy2(os.path.join(src_folder, f), os.path.join(dst_folder, f))

    # นับทั้งหมดหลัง copy
    all_files = os.listdir(dst_folder)
    total = len(all_files)
    print(f"รวมหลัง copy: {total}")

    # ถ้าเกิน MAX ให้ลบรูปเก่าออก (เก็บรูปใหม่ไว้)
    if total > MAX:
        new_names = set(renamed)
        old_in_dst = [f for f in all_files if f not in new_names]
        excess = total - MAX
        to_remove = random.sample(old_in_dst, excess)
        for f in to_remove:
            os.remove(os.path.join(dst_folder, f))
        print(f"ลบรูปเก่าออก: {excess} รูป")
        print(f"total สุดท้าย: {MAX} รูป")
    else:
        print(f"total สุดท้าย: {total} รูป (ไม่เกิน {MAX})")

    print("เสร็จแล้ว! OK")


if __name__ == "__main__":
    main()
