# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import cv2
# import os

# from sklearn.model_selection import GroupKFold
# from sklearn.metrics import roc_auc_score

# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from dataset import MelanomaDataset
# import timm





# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

# print("Using device:", device)

# import cv2
# from pathlib import Path
# import os
# from tqdm import tqdm

# def prepare_resized_images(input_dir, output_dir, size=(224, 224)):
#     input_dir = Path(input_dir)
#     output_dir = Path(output_dir)

#     if not input_dir.exists():
#         raise FileNotFoundError(f"Input folder not found: {input_dir}")

#     output_dir.mkdir(parents=True, exist_ok=True)

#     image_files = [
#         f for f in os.listdir(input_dir)
#         if f.lower().endswith((".jpg", ".jpeg", ".png"))
#     ]
#     print(f"Total images found: {len(image_files)}")

#     for img_name in tqdm(image_files):
#         input_path = str(input_dir / img_name)
#         output_path = str(output_dir / img_name)

#         if os.path.exists(output_path):
#             continue

#         try:
#             img = cv2.imread(input_path)
#             img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
#             cv2.imwrite(output_path, img)

#         except Exception as e:
#             print(f"Error with {img_name}: {e}")

#     print("Resized dataset ready.")

# DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))

# input_dir = DATASET_PATH / "jpeg" / "train"
# output_dir = DATASET_PATH / "jpeg_224" / "train"

# if not output_dir.exists() or len(os.listdir(output_dir)) == 0:
#     prepare_resized_images(input_dir, output_dir)
# else:
#     print("Resized dataset already exists.")


# train_path = output_dir
# labels_path = DATASET_PATH / "train.csv"
# df = pd.read_csv(labels_path)

# gkf = GroupKFold(n_splits=5)
# df["fold"] = -1

# for fold, (train_idx, val_idx) in enumerate(
#     gkf.split(df, df["target"], groups=df["patient_id"])
# ):
#     df.loc[val_idx, "fold"] = fold

# train_df = df[df.fold != 0]
# val_df = df[df.fold == 0]


# train_transforms = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.5),
#     A.ShiftScaleRotate(p=0.5),
#     A.Normalize(mean=(0.485, 0.456, 0.406),
#             std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ])

# val_transforms = A.Compose([
#     A.Normalize(mean=(0.485, 0.456, 0.406),
#             std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ])

# train_dataset = MelanomaDataset(train_df,str(train_path), train_transforms)
# val_dataset = MelanomaDataset(val_df, str(train_path), val_transforms)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)
# model = model.to(device)

# class_counts = train_df["target"].value_counts()


# pos_weight = torch.tensor([class_counts[0] / class_counts[1]], dtype=torch.float32).to(device)


# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# from tqdm import tqdm

# def train_one_epoch(model, loader):
#     model.train()
#     total_loss = 0

#     for images, labels in tqdm(loader):
#         images = images.to(device).float()
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)

#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     return total_loss / len(loader)



# def validate(model, loader):
#     model.eval()

#     preds = []
#     targets = []

#     with torch.no_grad():
#         for images, labels in loader:
#             images = images.to(device).float()
#             labels = labels.to(device)


#             outputs = model(images)
#             probs = torch.sigmoid(outputs).cpu().numpy()

#             preds.extend(probs.flatten())
#             targets.extend(labels.cpu().numpy().flatten())


#     auc = roc_auc_score(targets, preds)
#     return auc


# for epoch in range(5):
#     train_loss = train_one_epoch(model, train_loader)
#     val_auc = validate(model, val_loader)

#     print(f"Epoch {epoch}: Loss={train_loss:.4f}, AUC={val_auc:.4f}")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from dataset import MelanomaDataset
from baseline_model import get_model


# ========================
# DEVICE
# ========================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ========================
# PATHS
# ========================
DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))

train_path = DATASET_PATH / "jpeg_224" / "train"
labels_path = DATASET_PATH / "train.csv"


# ========================
# LOAD DATA
# ========================
df = pd.read_csv(labels_path)

gkf = GroupKFold(n_splits=5)
df["fold"] = -1

for fold, (train_idx, val_idx) in enumerate(
    gkf.split(df, df["target"], groups=df["patient_id"])
):
    df.loc[val_idx, "fold"] = fold

train_df = df[df.fold != 0].reset_index(drop=True)
val_df = df[df.fold == 0].reset_index(drop=True)


# ========================
# TRANSFORMS
# ========================
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transforms = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# ========================
# DATASETS
# ========================
train_dataset = MelanomaDataset(train_df, str(train_path), train_transforms)
val_dataset = MelanomaDataset(val_df, str(train_path), val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# ========================
# MODEL
# ========================
MODEL_NAME = "resnet50"   # 🔁 change to "efficientnet_b0" for final

model = get_model(MODEL_NAME)
model = model.to(device)


# ========================
# LOSS
# ========================
class_counts = train_df["target"].value_counts()

pos_weight = torch.tensor(
    [class_counts[0] / class_counts[1]],
    dtype=torch.float32
).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ========================
# OPTIMIZER
# ========================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# ========================
# TRAIN FUNCTION
# ========================
def train_one_epoch(model, loader):
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader):
        images = images.to(device).float()
        labels = labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ========================
# VALIDATION FUNCTION
# ========================
def validate(model, loader):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device).float()
            labels = labels.to(device).float()

            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            preds.extend(probs.flatten())
            targets.extend(labels.cpu().numpy().flatten())

    auc = roc_auc_score(targets, preds)
    return auc


# ========================
# TRAIN LOOP
# ========================
best_auc = 0

for epoch in range(3):  # baseline → 3 epochs
    train_loss = train_one_epoch(model, train_loader)
    val_auc = validate(model, val_loader)

    print(f"Epoch {epoch}: Loss={train_loss:.4f}, AUC={val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), f"{MODEL_NAME}_best.pth")
        print("✅ Saved best model")

print("Best AUC:", best_auc)
