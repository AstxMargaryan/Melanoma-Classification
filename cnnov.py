import os
import time
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from dataset import MelanomaDataset
from baseline_model import get_model
from preprocess import prepare_resized_images

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ======================
# DEVICE
# ======================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ======================
# PATHS
# ======================
DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))

input_dir = DATASET_PATH / "jpeg" / "train"
output_dir = DATASET_PATH / "jpeg_224" / "train"

os.makedirs("models", exist_ok=True)


# ======================
# PREPROCESS (resize once)
# ======================
if not output_dir.exists() or len(os.listdir(output_dir)) == 0:
    prepare_resized_images(input_dir, output_dir)
else:
    print("Resized dataset already exists.")


train_path = output_dir
labels_path = DATASET_PATH / "train.csv"


# ======================
# LOAD DATA
# ======================
df = pd.read_csv(labels_path)
df = df[["image_name", "patient_id", "target"]]


# ======================
# SPLIT (GroupKFold)
# ======================
gkf = GroupKFold(n_splits=5)
df["fold"] = -1

for fold, (train_idx, val_idx) in enumerate(
    gkf.split(df, df["target"], groups=df["patient_id"])
):
    df.loc[val_idx, "fold"] = fold

train_df = df[df.fold != 0].reset_index(drop=True)
val_df = df[df.fold == 0].reset_index(drop=True)


# ======================
# TRANSFORMS (minimal baseline)
# ======================
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# ======================
# DATASETS / LOADERS
# ======================
train_dataset = MelanomaDataset(train_df, train_path, train_transform)
val_dataset = MelanomaDataset(val_df, train_path, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# ======================
# VALIDATION
# ======================
def validate(model, val_loader, device):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            outputs = model(inputs).squeeze(1)
            probs = torch.sigmoid(outputs)

            preds.extend(probs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    if len(set(targets)) < 2:
        return 0.0

    return roc_auc_score(targets, preds)


# ======================
# TRAIN FUNCTION
# ======================
def train_model(model, train_loader, val_loader, device,
                epochs=5, lr=1e-3, save_path="models/cnn_baseline.pth"):

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()  # ✅ baseline (NO weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc = 0.0

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        print(f"\n🚀 Epoch [{epoch+1}/{epochs}]")

        for batch_idx, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)

        # validation
        val_auc = validate(model, val_loader, device)

        print(f"📊 Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print("💾 Saved best model")

    print(f"\n🔥 Best AUC: {best_auc:.4f}")

    return best_auc


# ======================
# RUN BASELINE
# ======================
model = get_model("cnn")  # 👈 SimpleCNN

print("\n====================")
print("Training Baseline CNN")
print("====================")

best_auc = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=5,
    lr=1e-3
)

print("\n📊 FINAL RESULT")
print(f"Baseline AUC: {best_auc:.4f}")
