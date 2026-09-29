"""
train_mobilenet.py
CNN-based image recognition: MobileNetV2 (torchvision, ImageNet pretrained)
ใช้รูป raw ไม่ผ่าน image processing (garbage_dataset_cls) — เทียบกับ no_processing ฝั่ง YOLO
ตัว MobileNetV2 เป็นคนละ framework จาก Ultralytics YOLO ที่ใช้อยู่ทั้งหมด เลยเขียน training loop เอง
ด้วย PyTorch/torchvision ตรงๆ

รัน:
    python train_mobilenet.py
โมเดล + confusion matrix จะถูกบันทึกที่: runs/classify/runs/garbage_cls_mobilenet_v2/
"""
import os, sys, time, copy
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "garbage_dataset_cls"  # รูป raw ไม่ผ่าน image processing
OUT_DIR = "runs/classify/runs/garbage_cls_mobilenet_raw"
EPOCHS = 150
PATIENCE = 30
BATCH_SIZE = 16
LR = 0.001
IMG_SIZE = 224  # ขนาดมาตรฐานที่ MobileNetV2 pretrain มา (ไม่ใช่ 640 แบบ YOLO)


def build_model(num_classes, device):
    """
    Transfer learning แบบคลาสสิก:
    - MobileNetV2 (pretrained ImageNet) เป็น CNN หลัก -> ล็อกน้ำหนักทั้งหมด (freeze) ไม่ให้ปรับ
    - ต่อท้ายด้วยชั้น Dense (Linear) ใหม่ที่เทรนได้อย่างเดียว เพื่อจำแนก 10 ประเภทขยะ
    - Softmax ไม่ได้ใส่เป็น layer แยกในโมเดล เพราะ nn.CrossEntropyLoss() รวม log-softmax ไว้ในตัวเองแล้ว
      (ใส่ Softmax ซ้อนก่อนเข้า CrossEntropyLoss จะทำให้เทรนผิดพลาดทางคณิตศาสตร์)
      ตอน predict จริงถึงค่อยเรียก softmax() แยกเพื่อแปลง logit เป็นความน่าจะเป็น/confidence %
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False  # freeze CNN หลักทั้งหมด (transfer learning)

    model.classifier[1] = nn.Linear(model.last_channel, num_classes)  # Dense ใหม่ (เทรนได้)
    return model.to(device)


def predict_proba(model, imgs):
    """ใช้ตอน predict จริง: logit -> softmax -> ความน่าจะเป็นแต่ละ class"""
    logits = model(imgs)
    return torch.softmax(logits, dim=1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"device: {device}")

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet stats (ต้องใช้ค่านี้คู่กับ pretrained weights)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.4),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tf)
    assert train_ds.classes == val_ds.classes, "train/val class ไม่ตรงกัน"
    classes = train_ds.classes
    print(f"classes ({len(classes)}): {classes}")
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(len(classes), device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"parameters: เทรนได้ {n_trainable:,} / ทั้งหมด {n_total:,} "
          f"({n_trainable/n_total*100:.1f}% — backbone ถูก freeze ไว้)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0005)  # อัปเดตแค่ชั้น dense ใหม่
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc, epochs_no_improve = 0.0, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            running_correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        scheduler.step()
        train_loss, train_acc = running_loss / total, running_correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / val_total
        dt = time.time() - t0

        print(f"epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}%  "
              f"val_acc={val_acc*100:.2f}%  ({dt:.1f}s)", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save({"model": copy.deepcopy(model.state_dict()), "classes": classes, "val_acc": best_acc},
                       os.path.join(OUT_DIR, "best.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"หยุดก่อนกำหนด (patience={PATIENCE}) ที่ epoch {epoch}")
                break

    print(f"\nเทรนเสร็จ! best val_acc = {best_acc*100:.2f}%")

    # ---------- โหลด best กลับมาทำ confusion matrix ----------
    ckpt = torch.load(os.path.join(OUT_DIR, "best.pt"), map_location=device)
    model = build_model(len(classes), device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)  # cm[pred][true]
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            for p, t in zip(preds, labels.numpy()):
                cm[p][t] += 1

    cm_norm = cm / cm.sum(axis=0, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(10, 9), dpi=200)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels(classes)
    ax.set_xlabel("True"); ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix Normalized (MobileNetV2 + Raw Image)")
    for i in range(n):
        for j in range(n):
            if cm_norm[i, j] > 0:
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_normalized.png"), facecolor="white")
    print(f"บันทึก confusion matrix: {OUT_DIR}/confusion_matrix_normalized.png")
    print(f"โมเดล: {OUT_DIR}/best.pt")


if __name__ == "__main__":
    main()
