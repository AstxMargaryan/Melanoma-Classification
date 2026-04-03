import os
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from dataset import MelanomaDataset
from baseline_model import get_model
from preprocess import prepare_resized_images

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


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
# OPTIONAL: SIMPLE GROUP SPLIT
# ======================
def create_group_split(df, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    df = df.copy()
    df["fold"] = -1

    for fold, (_, val_idx) in enumerate(
        gkf.split(df, df["target"], groups=df["patient_id"])
    ):
        df.loc[val_idx, "fold"] = fold

    return df


# ======================
# STRATIFIED PATIENT-WISE SPLIT
# ======================
def create_stratified_split(df, n_splits=5):
    patient_df = df.groupby("patient_id")["target"].max().reset_index()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    patient_df["fold"] = -1

    for fold, (_, val_idx) in enumerate(
        skf.split(patient_df["patient_id"], patient_df["target"])
    ):
        patient_df.loc[val_idx, "fold"] = fold

    df = df.merge(patient_df[["patient_id", "fold"]], on="patient_id", how="left")
    return df


# ======================
# TRANSFORMS
# ======================
train_transform_baseline = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

train_transform_final = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),

    A.Affine(
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        scale=(0.90, 1.10),
        rotate=(-20, 20),

        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.4
    ),

    A.OneOf([
        A.GaussianBlur(blur_limit=5),
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
    ], p=0.15),

    A.CoarseDropout(
        num_holes_range=(1, 6),
        hole_height_range=(8, 20),
        hole_width_range=(8, 20),
        fill=0,
        p=0.20
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# ======================
# SPLIT
# ======================
df = create_stratified_split(df, n_splits=5)

train_df = df[df["fold"] != 0].reset_index(drop=True)
val_df = df[df["fold"] == 0].reset_index(drop=True)

print("Train positives:", train_df["target"].sum())
print("Val positives:", val_df["target"].sum())
print("Train size:", len(train_df))
print("Val size:", len(val_df))

train_patients = set(train_df["patient_id"])
val_patients = set(val_df["patient_id"])
overlap = train_patients.intersection(val_patients)
print("Overlapping patients:", len(overlap))

for f in range(5):
    fold_df = df[df["fold"] == f]
    print(
        f"Fold {f} | size={len(fold_df)} | positives={fold_df['target'].sum()} | "
        f"positive_rate={fold_df['target'].mean():.4f}"
    )


# ======================
# CLASS IMBALANCE
# ======================
class_counts = train_df["target"].value_counts().sort_index()
pos_weight = class_counts[0] / class_counts[1]

print("Class counts:")
print(class_counts)
print(f"pos_weight: {pos_weight:.4f}")


# ======================
# Datasets
# ======================
cnn_train_dataset = MelanomaDataset(train_df, train_path, train_transform_baseline)
cnn_val_dataset = MelanomaDataset(val_df, train_path, val_transform)

resnet_train_dataset = MelanomaDataset(train_df, train_path, train_transform_final)
resnet_val_dataset = MelanomaDataset(val_df, train_path, val_transform)


# ======================
# Loaders
# ======================
cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=32, shuffle=True)
cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=32, shuffle=False)

resnet_train_loader = DataLoader(resnet_train_dataset, batch_size=32, shuffle=True)
resnet_val_loader = DataLoader(resnet_val_dataset, batch_size=32,shuffle=False)


# ======================
# VALIDATION
# ======================
def validate(model, val_loader, device, criterion):
    model.eval()

    val_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend(labels.cpu().numpy().flatten())

    val_loss /= max(len(val_loader), 1)

    if len(set(int(x) for x in all_targets)) < 2:
        auc = 0.0
    else:
        auc = roc_auc_score(all_targets, all_probs)

    return val_loss, auc, all_probs, all_targets


# ======================
# THRESHOLD TUNING
# ======================
def find_best_threshold(all_targets, all_probs):
    best_threshold = 0.5
    best_f1 = 0.0

    print("\nThreshold tuning results:")
    print("-" * 70)
    print(f"{'Threshold':<12}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}")
    print("-" * 70)

    for threshold in [i / 100 for i in range(10, 91, 5)]:
        preds = [1 if p >= threshold else 0 for p in all_probs]

        acc = accuracy_score(all_targets, preds)
        precision = precision_score(all_targets, preds, zero_division=0)
        recall = recall_score(all_targets, preds, zero_division=0)
        f1 = f1_score(all_targets, preds, zero_division=0)

        print(f"{threshold:<12.2f}{acc:<12.4f}{precision:<12.4f}{recall:<12.4f}{f1:<12.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print("-" * 70)
    print(f"✅ Best threshold by F1: {best_threshold:.2f}")
    print(f"✅ Best F1: {best_f1:.4f}")

    return best_threshold, best_f1


# ======================
# TRAINING
# ======================
def train_model(model, train_loader, val_loader, device, pos_weight,
                epochs=5, lr=1e-3, save_path="best_model.pth"):

    train_losses = []
    val_losses = []
    val_aucs = []

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        print("➡️ Running validation...")

        val_loss, val_auc, all_probs, all_targets = validate(
            model, val_loader, device, criterion
        )

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

    # threshold tuning on best saved model
    state_dict = torch.load(save_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)

    val_loss, val_auc, all_probs, all_targets = validate(
        model, val_loader, device, criterion
    )

    best_threshold, best_f1 = find_best_threshold(all_targets, all_probs)

    return train_losses, val_losses, val_aucs, best_threshold



model = get_model("cnn")

print("\n====================")
print("Training Baseline CNN")
print("====================")

train_losses, val_losses, val_aucs, cnn_best_threshold = train_model(
    model=model,
    train_loader=cnn_train_loader,
    val_loader=cnn_val_loader,
    device=device,
    pos_weight=pos_weight,
    epochs=5,
    lr=1e-3,
    save_path="models/baseline_cnn_best.pth"
)

print("\n📊 FINAL RESULT (CNN)")
print(f"Best AUC: {max(val_aucs):.4f}")
print(f"Best Threshold: {cnn_best_threshold:.2f}")


# import gc

# del model
# del train_losses, val_losses, val_aucs
# gc.collect()

# CUDA only (Colab)
if torch.cuda.is_available():
    torch.cuda.empty_cache()


model = get_model("resnet50")

print("\n====================")
print("Training ResNet50")
print("====================")

train_losses, val_losses, val_aucs, best_threshold = train_model(
    model=model,
    train_loader=resnet_train_loader,
    val_loader=resnet_val_loader,
    device=device,
    pos_weight=pos_weight,
    epochs=10,
    lr=1e-4,
    save_path="models/resnet50_final_best.pth"
)

print(f"\n🎯 Selected threshold: {best_threshold:.2f}")
print(f"📊 FINAL RESULT (ResNet50) | Best AUC: {max(val_aucs):.4f}")