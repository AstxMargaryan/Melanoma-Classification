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
from tqdm import tqdm

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
train_images = set(train_df["image_name"])
val_images = set(val_df["image_name"])

intersection = train_images.intersection(val_images)

print("Overlap:", len(intersection))



# ======================
# TRANSFORMS (minimal baseline)
# ======================
train_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
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

def validate(model, val_loader, device, criterion):
    model.eval()

    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)

            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(labels.cpu().numpy().flatten())

 
    val_loss /= max(len(val_loader), 1)


    # safer AUC
    if len(set(int(x) for x in all_targets)) < 2:
        auc = 0.0
    else:
        auc = roc_auc_score(all_targets, all_preds)

    return val_loss, auc


def train_model(model, train_loader, val_loader, device,pos_weight,
                epochs=5, lr=1e-3, save_path="best_model.pth"):

    train_losses = []
    val_losses = []
    val_aucs = []

    model = model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc = 0.0

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0


        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):

            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # ======================
        # VALIDATION
        # ======================
        print("➡️ Running validation...")

        val_loss, val_auc = validate(model, val_loader, device, criterion)

        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        print(
            f"📊 Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print("💾 Best model saved")

    print(f"\n🔥 Best AUC: {best_auc:.4f}")

    return train_losses, val_losses, val_aucs

class_counts = train_df["target"].value_counts()
pos_weight = class_counts[0] / class_counts[1]

# ======================
# TRAIN BASELINE CNN
# ======================

model = get_model("cnn")

print("\n====================")
print("Training Baseline CNN")
print("====================")

train_losses, val_losses, val_aucs = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    pos_weight=pos_weight,
    epochs=5,
    lr=1e-3,
    save_path="models/baseline_cnn_best.pth"
)

print("\n📊 FINAL RESULT")
print(f"Best AUC: {max(val_aucs):.4f}")

