import os
import shutil
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from dataset import MelanomaDataset
from baseline_model import get_model
from preprocess import prepare_resized_images


# =====================================================
# 1. REPRODUCIBILITY
# =====================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# =====================================================
# 2. DEVICE
# =====================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# =====================================================
# 3. PATHS
# =====================================================
DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))

# original SIIM-ISIC 2020 images
input_dir = DATASET_PATH / "jpeg" / "train"

# final training folder (all resized images will live here)
output_dir = DATASET_PATH / "jpeg_224" / "train"

# main train csv
labels_path = DATASET_PATH / "train.csv"

# external dataset csv and images
external_labels_path = DATASET_PATH / "train-metadata.csv"
external_img_dir = DATASET_PATH / "image"

os.makedirs("models", exist_ok=True)


# =====================================================
# 4. PREPROCESS ORIGINAL DATA (resize once)
# =====================================================
if not output_dir.exists() or len(os.listdir(output_dir)) == 0:
    prepare_resized_images(input_dir, output_dir)
else:
    print("Resized dataset already exists.")

train_path = output_dir


# =====================================================
# 5. LOAD ORIGINAL DATA
# =====================================================
df = pd.read_csv(labels_path)
df = df[["image_name", "patient_id", "target"]]

print("\nOriginal main df:")
print(df.head())
print("Original shape:", df.shape)
print("Original positives:", df["target"].sum())


# =====================================================
# 6. LOAD EXTERNAL DATA
# =====================================================
external_df = pd.read_csv(external_labels_path)

print("\nRaw external df:")
print(external_df.head())


# external file names are mixed:
# some are ISIC_xxx.jpg
# some are ISIC_xxx_downsampled.jpg
def find_image_name(isic_id, img_dir):
    if (img_dir / f"{isic_id}.jpg").exists():
        return isic_id
    elif (img_dir / f"{isic_id}_downsampled.jpg").exists():
        return isic_id + "_downsampled"
    else:
        return None


external_df["image_name"] = external_df["isic_id"].apply(
    lambda x: find_image_name(x, external_img_dir)
)

# keep only rows where matching image exists
external_df = external_df.dropna(subset=["image_name"])

# keep only needed columns
external_df = external_df[["image_name", "patient_id", "target"]]

print("\nProcessed external df:")
print(external_df.head())
print("External shape:", external_df.shape)
print("External positives:", external_df["target"].sum())


# =====================================================
# 7. COPY EXTERNAL IMAGES INTO FINAL TRAIN FOLDER
# =====================================================
# we want all train images (original + external) available in one folder:
# DATASET_PATH / jpeg_224 / train

copied_count = 0
for img in external_img_dir.glob("*.jpg"):
    target_path = output_dir / img.name
    if not target_path.exists():
        shutil.copy(img, target_path)
        copied_count += 1

print("\nCopied external images:", copied_count)

# verify one sample external image exists in final train folder
sample_name = external_df.iloc[0]["image_name"]
sample_path = output_dir / f"{sample_name}.jpg"
print("Sample external image path:", sample_path)
print("Exists:", sample_path.exists())


# =====================================================
# 8. CREATE STRATIFIED PATIENT-WISE FOLDS
# =====================================================
def create_stratified_split(df, n_splits=5):
    """
    Split original SIIM-ISIC 2020 data only.
    One patient must belong to exactly one fold.
    Stratification is done at patient level:
    if a patient has at least one melanoma image -> patient target = 1
    """
    patient_df = df.groupby("patient_id")["target"].max().reset_index()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    patient_df["fold"] = -1

    for fold, (_, val_idx) in enumerate(
        skf.split(patient_df["patient_id"], patient_df["target"])
    ):
        patient_df.loc[val_idx, "fold"] = fold

    df = df.merge(patient_df[["patient_id", "fold"]], on="patient_id", how="left")
    return df


df = create_stratified_split(df, n_splits=5)

print("\nFold statistics on ORIGINAL data:")
for f in range(5):
    fold_df = df[df["fold"] == f]
    print(
        f"Fold {f} | "
        f"size={len(fold_df)} | "
        f"positives={fold_df['target'].sum()} | "
        f"positive_rate={fold_df['target'].mean():.4f}"
    )


# =====================================================
# 9. PREPARE TRAIN/VAL FOR ONE FOLD
# =====================================================
def prepare_fold_data(df, external_df, fold_num):
    """
    For a given fold:
    - validation = original fold only
    - training = remaining original folds + ALL external data
    """
    train_df = df[df["fold"] != fold_num].reset_index(drop=True)
    val_df = df[df["fold"] == fold_num].reset_index(drop=True)

    # external data goes ONLY into train
    train_df = pd.concat([train_df, external_df], ignore_index=True)

    # recompute pos_weight from merged train set
    class_counts = train_df["target"].value_counts().sort_index()
    pos_weight = class_counts[0] / class_counts[1]

    print(f"\nFold {fold_num} summary:")
    print("Train size:", len(train_df))
    print("Train positives:", train_df["target"].sum())
    print("Val size:", len(val_df))
    print("Val positives:", val_df["target"].sum())
    print("pos_weight:", round(pos_weight, 4))

    # verify no patient overlap in original split part
    train_patients = set(df[df["fold"] != fold_num]["patient_id"])
    val_patients = set(df[df["fold"] == fold_num]["patient_id"])
    overlap = train_patients.intersection(val_patients)
    print("Original patient overlap:", len(overlap))

    return train_df, val_df, pos_weight


# =====================================================
# 10. AUGMENTATIONS
# =====================================================
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
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])


# =====================================================
# 11. DATALOADERS
# =====================================================
def create_dataloaders(train_df, val_df, train_transform, val_transform, train_path, batch_size=32):
    train_dataset = MelanomaDataset(train_df, train_path, train_transform)
    val_dataset = MelanomaDataset(val_df, train_path, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


# =====================================================
# 12. VALIDATION
# =====================================================
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


# =====================================================
# 13. THRESHOLD TUNING
# =====================================================
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


# =====================================================
# 14. TRAIN ONE MODEL FOR ONE FOLD
# =====================================================
def train_model(
    model,
    train_loader,
    val_loader,
    device,
    pos_weight,
    epochs=8,
    lr=1e-4,
    save_path="best_model.pth",
    patience=2
):
    train_losses = []
    val_losses = []
    val_aucs = []

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc = 0.0
    epochs_without_improvement = 0

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
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"⏳ No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= patience:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs")
            break

    print(f"\n🔥 Best AUC: {best_auc:.4f}")

    # load best checkpoint
    state_dict = torch.load(save_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)

    # final validation on best model for threshold tuning
    val_loss, val_auc, all_probs, all_targets = validate(
        model, val_loader, device, criterion
    )

    best_threshold, best_f1 = find_best_threshold(all_targets, all_probs)

    return train_losses, val_losses, val_aucs, best_threshold


def run_5fold_training(
    model_name,
    df,
    external_df,
    train_path,
    train_transform,
    val_transform,
    device,
    epochs=8,
    lr=1e-4,
    batch_size=32,
    patience=2
):
    all_fold_aucs = []
    all_fold_thresholds = []

    print("\n" + "=" * 60)
    print(f"STARTING 5-FOLD TRAINING FOR: {model_name.upper()}")
    print("=" * 60)

    for fold_num in range(5):
        print("\n" + "=" * 50)
        print(f"TRAINING {model_name.upper()} | FOLD {fold_num}")
        print("=" * 50)

        train_df, val_df, pos_weight = prepare_fold_data(df, external_df, fold_num)

        train_loader, val_loader = create_dataloaders(
            train_df=train_df,
            val_df=val_df,
            train_transform=train_transform,
            val_transform=val_transform,
            train_path=train_path,
            batch_size=batch_size
        )

        model = get_model(model_name)
        save_path = f"models/{model_name}_fold{fold_num}.pth"

        train_losses, val_losses, val_aucs, best_threshold = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            pos_weight=pos_weight,
            epochs=epochs,
            lr=lr,
            save_path=save_path,
            patience=patience
        )

        fold_best_auc = max(val_aucs)

        all_fold_aucs.append(fold_best_auc)
        all_fold_thresholds.append(best_threshold)

        print(f"\n✅ {model_name} Fold {fold_num} best AUC: {fold_best_auc:.4f}")
        print(f"✅ {model_name} Fold {fold_num} best threshold: {best_threshold:.2f}")

        del model

    print("\n" + "=" * 50)
    print(f"{model_name.upper()} FINAL 5-FOLD RESULTS")
    print("=" * 50)

    for i, (auc, thr) in enumerate(zip(all_fold_aucs, all_fold_thresholds)):
        print(f"Fold {i}: AUC = {auc:.4f} | Threshold = {thr:.2f}")

    print(f"\nMean AUC: {np.mean(all_fold_aucs):.4f}")
    print(f"Std AUC: {np.std(all_fold_aucs):.4f}")
    print(f"Mean Threshold: {np.mean(all_fold_thresholds):.2f}")

    return all_fold_aucs, all_fold_thresholds





resnet_aucs, resnet_thresholds = run_5fold_training(
    model_name="resnet50",
    df=df,
    external_df=external_df,
    train_path=train_path,
    train_transform=train_transform_final,
    val_transform=val_transform,
    device=device,
    epochs=8,
    lr=1e-4,
    batch_size=32,
    patience=2
)

# b3_aucs, b3_thresholds = run_5fold_training(
#     model_name="efficientnet_b3",
#     df=df,
#     external_df=external_df,
#     train_path=train_path,
#     train_transform=train_transform_final,
#     val_transform=val_transform,
#     device=device,
#     epochs=10,
#     lr=1e-4,
#     batch_size=32
# )

# b4_aucs, b4_thresholds = run_5fold_training(
#     model_name="efficientnet_b4",
#     df=df,
#     external_df=external_df,
#     train_path=train_path,
#     train_transform=train_transform_final,
#     val_transform=val_transform,
#     device=device,
#     epochs=10,
#     lr=1e-4,
#     batch_size=32
# )