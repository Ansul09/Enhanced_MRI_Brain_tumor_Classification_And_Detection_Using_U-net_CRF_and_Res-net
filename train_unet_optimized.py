import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_loader_2d import BrainTumorDataset
from unet_model import UNet

# -----------------------------
# DEVICE
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

print("🚀 Using device:", DEVICE)
if USE_AMP:
    print("🟢 GPU:", torch.cuda.get_device_name(0))
else:
    print("🔴 WARNING: GPU NOT AVAILABLE")

# -----------------------------
# PATHS (⚠️ USE LOCAL COLAB STORAGE)
# -----------------------------
DATA_DIR = "/content/dataset_2d"   # 🔥 MUST be local, not Drive
MODEL_DIR = "/content/models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "unet_trained.pth")

# -----------------------------
# HYPERPARAMETERS (FAST & SAFE)
# -----------------------------
BATCH_SIZE = 8          # 🔥 faster
EPOCHS = 25
LR = 3e-5
NUM_WORKERS = 2         # 🔥 do NOT increase

# -----------------------------
# DATASETS
# -----------------------------
train_dataset = BrainTumorDataset(DATA_DIR, split="train")
val_dataset   = BrainTumorDataset(DATA_DIR, split="val")

print(f"✅ TRAIN slices: {len(train_dataset)}")
print(f"✅ VAL slices  : {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP
)

# -----------------------------
# MODEL
# -----------------------------
model = UNet(in_channels=1, out_channels=1).to(DEVICE)
print("🆕 Training from scratch")

# -----------------------------
# LOSS
# -----------------------------
pos_weight = torch.tensor([8.0], device=DEVICE)
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.Adam(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    denom = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2 * intersection + smooth) / (denom + smooth)).mean()

def combined_loss(pred, target):
    return bce(pred, target) + dice_loss(pred, target)

# =====================================================
# 🔍 STEP-2 DEBUG: SINGLE BATCH TIMING CHECK
# =====================================================
print("\n🔍 DEBUG: Measuring ONE batch time...")
model.train()

start = time.time()
imgs, masks = next(iter(train_loader))

imgs = imgs.to(DEVICE)
masks = masks.unsqueeze(1).to(DEVICE).float()

optimizer.zero_grad()
with torch.cuda.amp.autocast(enabled=USE_AMP):
    preds = model(imgs)
    loss = combined_loss(preds, masks)

loss.backward()
optimizer.step()

print(f"⏱ One batch time: {time.time() - start:.3f} seconds")

print("🔍 DEBUG COMPLETE\n")
# =====================================================

# -----------------------------
# TRAINING LOOP
# -----------------------------
for epoch in range(EPOCHS):
    print(f"\n🧠 Epoch {epoch+1}/{EPOCHS}")
    model.train()

    epoch_start = time.time()
    train_loss = 0.0
    batches = 0

    for imgs, masks in train_loader:
        imgs = imgs.to(DEVICE)
        masks = masks.unsqueeze(1).to(DEVICE).float()

        if masks.sum() == 0:
            continue

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            preds = model(imgs)
            loss = combined_loss(preds, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        batches += 1

    train_loss /= max(batches, 1)

    print(f"📉 Train Loss: {train_loss:.4f}")
    print(f"⏱ Epoch time: {(time.time() - epoch_start)/60:.2f} minutes")

    torch.save(model.state_dict(), MODEL_PATH)
    print("💾 Model saved")

print("\n✅ Training finished")
