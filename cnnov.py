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
# DATA PATHS
# ======================

DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))

input_dir = DATASET_PATH / "jpeg" / "train"
output_dir = DATASET_PATH / "jpeg_224" / "train"

# create model folder
os.makedirs("models", exist_ok=True)


# ======================
# PREPARE RESIZED DATASET
# ======================

if not output_dir.exists() or len(os.listdir(output_dir)) == 0:
    prepare_resized_images(input_dir, output_dir)
else:
    print("Resized dataset already exists.")


train_path = output_dir
labels_path = DATASET_PATH / "train.csv"


# ======================
# LOAD CSV
# ======================

df = pd.read_csv(labels_path)
df = df[["image_name", "patient_id", "target"]]


# ======================
# GROUP K-FOLD
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
# TRANSFORMS
# ======================

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])


# ======================
# DATASETS
# ======================

train_dataset = MelanomaDataset(train_df, train_path, train_transform)
val_dataset = MelanomaDataset(val_df, train_path, val_transform)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# ======================
# VALIDATION
# ======================

def validate(model, val_loader, device, criterion):

    model.eval()

    preds = []
    targets = []
    val_loss = 0.0

    print("🔍 Validation started...")

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.to(device)
            labels = labels.to(device).float()

            outputs = model(inputs).squeeze(1)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)

            preds.extend(probs.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())

    val_loss /= len(val_loader)

    # safe AUC calculation
    if len(set(targets)) < 2:
        auc = 0.0
    else:
        auc = roc_auc_score(targets, preds)

    print("✅ Validation done")

    return val_loss, auc

class_counts = train_df["target"].value_counts()
pos_weight = class_counts[0] / class_counts[1]


# ======================
# TRAIN FUNCTION
# ======================

def train_model(model, train_loader, val_loader, device,
                epochs=5, lr=1e-3, save_path="models/baseline_cnn_best.pth"):

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_aucs = []

    best_auc = 0.0

    start_time = time.time()

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        print(f"\n🚀 Epoch [{epoch+1}/{epochs}]")

        # ======================
        # TRAIN
        # ======================

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
                print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)

        train_losses.append(avg_loss)

        # ======================
        # VALIDATION
        # ======================

        print("➡️ Running validation...")

        val_loss, val_auc = validate(model, val_loader, device, criterion)

        val_aucs.append(val_auc)

        print(
            f"📊 Train Loss: {avg_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        # ======================
        # SAVE BEST MODEL
        # ======================

        if val_auc > best_auc:

            best_auc = val_auc

            torch.save(model.state_dict(), save_path)

            print("💾 Best model saved")

    print(f"\n⏱ Training finished in {time.time() - start_time:.2f}s")

    print(f"🔥 Best AUC: {best_auc:.4f}")

    return train_losses, val_aucs


# ======================
# BASELINE TRAINING
# ======================

baseline_model = get_model("baseline_cnn")

print("\n====================")
print("Training Baseline CNN")
print("====================")

baseline_losses, baseline_aucs = train_model(
    model=baseline_model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=5,
    lr=1e-3,
    save_path="models/baseline_cnn_best.pth"
)


print("\n📊 BASELINE RESULT")
print(f"Baseline CNN Best AUC: {max(baseline_aucs):.4f}")