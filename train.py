import os
import shutil
import random
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt 
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
from model import get_model
from preprocess import prepare_resized_images


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)



DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))

# original SIIM-ISIC 2020 images
input_dir = DATASET_PATH / "jpeg" / "train"

# final training folder (all resized images will live here)
output_dir = DATASET_PATH / "jpeg_224" / "train"
output_dir2 = DATASET_PATH / "jpeg_300" / "train" 

# main train csv
labels_path = DATASET_PATH / "train.csv"

# external dataset csv and images
external_labels_path = DATASET_PATH / "train-metadata.csv"
external_img_dir = DATASET_PATH / "image"

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)


if not output_dir.exists() or len(os.listdir(output_dir)) == 0:
    prepare_resized_images(input_dir, output_dir)
else:
    print("Resized dataset already exists.")


train_path = output_dir

df = pd.read_csv(labels_path)
df = df[["image_name", "patient_id", "target"]]

print("\nOriginal main df:")
print(df.head())
print("Original shape:", df.shape)
print("Original positives:", df["target"].sum())


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



def prepare_fold_data(df, external_df, fold_num):
    """
    For a given fold:
    - validation = original fold only
    - training = remaining original folds + ALL external data
    """

    
    # ORIGINAL TRAIN/VAL SPLIT
    original_train_df = df[df["fold"] != fold_num].reset_index(drop=True)
    val_df = df[df["fold"] == fold_num].reset_index(drop=True)

    # original train stats
    orig_class_counts = original_train_df["target"].value_counts().sort_index()
    orig_pos_count = int(original_train_df["target"].sum())
    orig_pos_rate = original_train_df["target"].mean()

    if 0 in orig_class_counts and 1 in orig_class_counts:
        orig_pos_weight = orig_class_counts[0] / orig_class_counts[1]
    else:
        orig_pos_weight = None

    
    # MERGE EXTERNAL DATA
    train_df = pd.concat([original_train_df, external_df], ignore_index=True)

    # merged train stats
    merged_class_counts = train_df["target"].value_counts().sort_index()
    merged_pos_count = int(train_df["target"].sum())
    merged_pos_rate = train_df["target"].mean()

    if 0 in merged_class_counts and 1 in merged_class_counts:
        pos_weight = merged_class_counts[0] / merged_class_counts[1]
    else:
        pos_weight = None

    
    # PRINT COMPARISON
    print(f"\nFold {fold_num} summary:")

    print("Before external data:")
    print("  Train size:", len(original_train_df))
    print("  Positive samples:", orig_pos_count)
    print(f"  Positive rate: {orig_pos_rate:.4f}")
    if orig_pos_weight is not None:
        print(f"  pos_weight: {orig_pos_weight:.4f}")

    print("\nAfter external data:")
    print("  Train size:", len(train_df))
    print("  Positive samples:", merged_pos_count)
    print(f"  Positive rate: {merged_pos_rate:.4f}")
    if pos_weight is not None:
        print(f"  pos_weight: {pos_weight:.4f}")

    print("\nValidation:")
    print("  Val size:", len(val_df))
    print("  Val positives:", int(val_df["target"].sum()))
    print(f"  Val positive rate: {val_df['target'].mean():.4f}")

    # verify no patient overlap in original split part
    train_patients = set(df[df["fold"] != fold_num]["patient_id"])
    val_patients = set(df[df["fold"] == fold_num]["patient_id"])
    overlap = train_patients.intersection(val_patients)
    print("\nOriginal patient overlap:", len(overlap))

    return train_df, val_df, pos_weight


train_transform_baseline = A.Compose([
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
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
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])


def create_dataloaders(train_df, val_df, train_transform, val_transform, train_path, batch_size=32):
    train_dataset = MelanomaDataset(train_df, train_path, train_transform)
    val_dataset = MelanomaDataset(val_df, train_path, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader


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


def find_best_threshold_f2(all_targets, all_probs):
    best_threshold = 0.5
    best_f2 = 0.0

    print("\nThreshold tuning (F2-based):")
    print("-" * 80)
    print(f"{'Thr':<8}{'Acc':<10}{'Prec':<10}{'Recall':<10}{'F1':<10}{'F2':<10}")
    print("-" * 80)

    for threshold in [i / 100 for i in range(10, 91, 5)]:
        preds = [1 if p >= threshold else 0 for p in all_probs]

        acc = accuracy_score(all_targets, preds)
        precision = precision_score(all_targets, preds, zero_division=0)
        recall = recall_score(all_targets, preds, zero_division=0)
        f1 = f1_score(all_targets, preds, zero_division=0)

        # F2 calculation
        if precision + recall == 0:
            f2 = 0
        else:
            f2 = (5 * precision * recall) / (4 * precision + recall)

        print(f"{threshold:<8.2f}{acc:<10.4f}{precision:<10.4f}{recall:<10.4f}{f1:<10.4f}{f2:<10.4f}")

        if f2 > best_f2:
            best_f2 = f2
            best_threshold = threshold

    
    print(f"Best threshold (F2): {best_threshold:.2f}")
    print(f"Best F2: {best_f2:.4f}")

    return best_threshold


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    pos_weight,
    epochs=8,
    lr=1e-4,
    save_path="best_model.pth",
    patience=2,
    use_scheduler=False
):
    train_losses = []
    val_losses = []
    val_aucs = []

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.3,
            patience=1
        )

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

        print("Running validation...")

        val_loss, val_auc, all_probs, all_targets = validate(
            model, val_loader, device, criterion
        )

        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        if scheduler is not None:
            scheduler.step(val_auc)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print("Best model saved")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print(f"\n Best AUC: {best_auc:.4f}")

    # load best checkpoint
    state_dict = torch.load(save_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)

    # final validation on best model for threshold tuning
    val_loss, val_auc, all_probs, all_targets = validate(
        model, val_loader, device, criterion
    )

    best_threshold = find_best_threshold_f2(all_targets, all_probs)

    return train_losses, val_losses, val_aucs, best_threshold


def run_kfold_training(
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
    patience=2,
    n_folds=1,
    use_scheduler=False
):
    all_fold_aucs = []
    all_fold_thresholds = []
    all_fold_train_losses = []
    all_fold_val_losses = []
    all_fold_val_aucs = []

    print(f"\nSTARTING TRAINING FOR: {model_name.upper()}")

    for fold_num in range(n_folds):
        print(f"\nTRAINING {model_name.upper()} | FOLD {fold_num}")

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
            patience=patience,
            use_scheduler=use_scheduler
        )

        fold_best_auc = max(val_aucs)

        all_fold_aucs.append(fold_best_auc)
        all_fold_thresholds.append(best_threshold)
        all_fold_train_losses.append(train_losses)
        all_fold_val_losses.append(val_losses)
        all_fold_val_aucs.append(val_aucs)

        print(f"\n {model_name} Fold {fold_num} best AUC: {fold_best_auc:.4f}")
        print(f" {model_name} Fold {fold_num} best threshold: {best_threshold:.2f}")

        del model

    
    print(f"{model_name.upper()} RESULTS")
   

    for i, (auc, thr) in enumerate(zip(all_fold_aucs, all_fold_thresholds)):
        print(f"Fold {i}: AUC = {auc:.4f} | Threshold = {thr:.2f}")

    print(f"\nMean AUC: {np.mean(all_fold_aucs):.4f}")
    print(f"Std AUC: {np.std(all_fold_aucs):.4f}")
    print(f"Mean Threshold: {np.mean(all_fold_thresholds):.2f}")

    return {
        "fold_aucs": all_fold_aucs,
        "fold_thresholds": all_fold_thresholds,
        "train_losses": all_fold_train_losses,
        "val_losses": all_fold_val_losses,
        "val_auc_curves": all_fold_val_aucs,
    }


def plot_training_curves(results, model_name):
    train_losses = results["train_losses"][0]
    val_losses = results["val_losses"][0]
    val_aucs = results["val_auc_curves"][0]

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Training vs Validation Loss")
    plt.legend()

    plt.savefig(f"plots/{model_name}_loss.png") 
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(val_aucs, label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title(f"{model_name} - Validation AUC")
    plt.legend()

    plt.savefig(f"plots/{model_name}_auc.png")  
    plt.close()



cnn_results = run_kfold_training(
    model_name="cnn",
    df=df,
    external_df=external_df,
    train_path=train_path,
    train_transform=train_transform_baseline,
    val_transform=val_transform,
    device=device,
    epochs=3,
    lr=1e-3,
    batch_size=32,
    patience=2,
    n_folds=1,
    use_scheduler=False
)


resnet_results = run_kfold_training(
    model_name="resnet50",
    df=df,
    external_df=external_df,
    train_path=train_path,
    train_transform=train_transform_final,
    val_transform=val_transform,
    device=device,
    epochs=10,
    lr=1e-4,
    batch_size=32,
    patience=3,
    n_folds=1,
    use_scheduler=True
)


b3_results = run_kfold_training(
    model_name="efficientnet_b3",
    df=df,
    external_df=external_df,
    train_path=train_path,
    train_transform=train_transform_final,
    val_transform=val_transform,
    device=device,
    epochs=10,
    lr=1e-4,
    batch_size=32,
    patience=3,
    n_folds=1,
    use_scheduler=True
)

plot_training_curves(cnn_results, "Baseline CNN")
plot_training_curves(resnet_results, "ResNet50")
plot_training_curves(b3_results, "EfficientNet-B3")



def summarize_model_results(name, results):
    aucs = results["fold_aucs"]
    thrs = results["fold_thresholds"]

    print(f"\n{name}")
    print(f"  Fold AUCs: {[round(x, 4) for x in aucs]}")
    print(f"  Mean AUC: {np.mean(aucs):.4f}")
    print(f"  Std AUC:  {np.std(aucs):.4f}")
    print(f"  Mean Threshold: {np.mean(thrs):.4f}")



print("MODEL SUMMARY")

summarize_model_results("Baseline CNN", cnn_results)
summarize_model_results("ResNet50", resnet_results)
summarize_model_results("EfficientNet-B3", b3_results)


cnn_mean_auc = np.mean(cnn_results["fold_aucs"])
resnet_mean_auc = np.mean(resnet_results["fold_aucs"])
b3_mean_auc = np.mean(b3_results["fold_aucs"])


print("IMPROVEMENT VS BASELINE CNN")
print(f"ResNet50  - CNN: +{resnet_mean_auc - cnn_mean_auc:.4f}")
print(f"EffNet-B3 - CNN: +{b3_mean_auc - cnn_mean_auc:.4f}")


model_scores = {
    "cnn": cnn_mean_auc,
    "resnet50": resnet_mean_auc,
    "efficientnet_b3": b3_mean_auc,
}

best_model_name = max(model_scores, key=model_scores.get)
best_model_mean_auc = model_scores[best_model_name]

if best_model_name == "cnn":
    best_threshold = float(np.mean(cnn_results["fold_thresholds"]))
    best_checkpoint = "models/cnn_fold0.pth"
elif best_model_name == "resnet50":
    best_threshold = float(np.mean(resnet_results["fold_thresholds"]))
    best_checkpoint = "models/resnet50_fold0.pth"
elif best_model_name == "efficientnet_b3":
    best_threshold = float(np.mean(b3_results["fold_thresholds"]))
    best_checkpoint = "models/efficientnet_b3_fold0.pth"


print("BEST SINGLE MODEL")
print(f"Best model: {best_model_name}")
print(f"Best mean AUC: {best_model_mean_auc:.4f}")
print(f"Best threshold: {best_threshold:.4f}")
print(f"Best checkpoint: {best_checkpoint}")

best_model_config = {
    "model_name": best_model_name,
    "checkpoint": best_checkpoint,
    "threshold": float(best_threshold),
    "best_auc": float(best_model_mean_auc)
}

config_path = "models/best_model_config.json"

with open(config_path, "w") as f:
    json.dump(best_model_config, f, indent=4)

print(f"\n Best model config saved to: {config_path}")