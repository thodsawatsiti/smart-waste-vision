"""
eval_hardval.py
เทียบโมเดลเก่า (best.pt เดิม) vs โมเดลใหม่ (fine-tuned) บนชุด hard_val/
ซึ่งเป็นรูปถือมือจริงที่ "ไม่เคย" ถูกเอาไปเทรนเลย (holdout)
"""
import os, sys, io, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from ultralytics import YOLO

OLD_MODEL = "runs/classify/runs/garbage_cls_no_processing_v2/weights/best.pt"
NEW_MODEL = "runs/classify/runs/classify/runs/garbage_cls_hardexamples_v1/weights/best.pt"
HARD_VAL_DIR = "hard_val"

def eval_model(model_path, label):
    model = YOLO(model_path)
    classes = sorted(d for d in os.listdir(HARD_VAL_DIR) if os.path.isdir(os.path.join(HARD_VAL_DIR, d)))
    total, correct = 0, 0
    per_class = {}
    wrong_details = []
    for cls in classes:
        paths = glob.glob(os.path.join(HARD_VAL_DIR, cls, "*"))
        c_tot, c_cor = 0, 0
        for p in paths:
            res = model.predict(p, verbose=False)[0]
            pred = model.names[int(res.probs.top1)]
            ok = int(pred == cls)
            c_tot += 1; c_cor += ok
            total += 1; correct += ok
            if not ok:
                wrong_details.append((cls, pred, os.path.basename(p)))
        per_class[cls] = (c_cor, c_tot)

    print(f"\n=== {label} ===")
    for cls, (c, t) in per_class.items():
        print(f"  {cls}: {c}/{t}")
    print(f"  รวม: {correct}/{total} = {correct/total*100:.1f}%")
    if wrong_details:
        print("  ทายผิด:")
        for true_c, pred_c, fname in wrong_details:
            print(f"    true={true_c} pred={pred_c}  {fname}")
    return correct, total

if __name__ == "__main__":
    old_c, old_t = eval_model(OLD_MODEL, "โมเดลเดิม (ก่อน fine-tune)")
    new_c, new_t = eval_model(NEW_MODEL, "โมเดลใหม่ (หลัง fine-tune)")
    print(f"\n=== สรุป ===")
    print(f"  เดิม:  {old_c}/{old_t} = {old_c/old_t*100:.1f}%")
    print(f"  ใหม่:  {new_c}/{new_t} = {new_c/new_t*100:.1f}%")
