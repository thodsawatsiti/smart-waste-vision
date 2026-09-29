"""
finetune_hard_examples.py
Fine-tune ต่อจาก best.pt เดิม (ไม่ใช่เทรนใหม่ตั้งแต่ต้น) ด้วย epoch น้อย + lr ต่ำ
เป้าหมาย: ให้โมเดลเรียนรู้ท่าถือมือ (hand-held) ที่เพิ่งเสริมเข้าไป
          โดยไม่ลืมของเดิมที่แม่นอยู่แล้ว (catastrophic forgetting)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from ultralytics import YOLO

BASE_MODEL = "runs/classify/runs/garbage_cls_no_processing_v2/weights/best.pt"
DATASET = "garbage_dataset_cls"
RUN_NAME = "garbage_cls_hardexamples_v1"

def main():
    model = YOLO(BASE_MODEL)
    model.train(
        data=DATASET,
        epochs=15,
        lr0=0.0005,      # lr ต่ำกว่าเทรนปกติมาก เพราะเป็นการปรับ ไม่ใช่เทรนจากศูนย์
        patience=15,
        device=0,
        project="runs/classify/runs",
        name=RUN_NAME,
        exist_ok=True,
        verbose=True,
    )
    print(f"\nเสร็จแล้ว! โมเดลใหม่: runs/classify/runs/{RUN_NAME}/weights/best.pt")

if __name__ == "__main__":
    main()
