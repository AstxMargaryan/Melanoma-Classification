
# import math
# import math
# import os
# import pandas as pd
# import torch
# import torch.nn as nn
# from pathlib import Path
# from torch.utils.data import DataLoader
# from sklearn.metrics import  roc_auc_score
# from dataset import MelanomaDataset
# from baseline_model import  get_model

# from sklearn.model_selection import GroupKFold
# from preprocess import prepare_resized_images
# import albumentations as A
# from albumentations.pytorch import ToTensorV2


# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

# print("Using device:", device)


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
# df = df[["image_name", "patient_id", "target"]]

# gkf = GroupKFold(n_splits=5)
# df["fold"] = -1

# for fold, (train_idx, val_idx) in enumerate(
#     gkf.split(df, df["target"], groups=df["patient_id"])
# ):
#     df.loc[val_idx, "fold"] = fold

# train_df = df[df.fold != 0].reset_index(drop=True)
# val_df = df[df.fold == 0].reset_index(drop=True)



# train_transform = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.5),
#     A.ShiftScaleRotate(
#         shift_limit=0.05,
#         scale_limit=0.1,
#         rotate_limit=15,
#         p=0.5
#     ),
#     A.Normalize(
#         mean=(0.485, 0.456, 0.406),
#         std=(0.229, 0.224, 0.225)
#     ),
#     ToTensorV2()
# ])


# val_transform = A.Compose([
#     A.Normalize(
#         mean=(0.485, 0.456, 0.406),
#         std=(0.229, 0.224, 0.225)
#     ),
#     ToTensorV2()
# ])


# train_dataset = MelanomaDataset(train_df, train_path, train_transform)
# val_dataset = MelanomaDataset(val_df, train_path, val_transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=2,pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# def validate(model, val_loader, device):
#     model.eval()

#     preds = []
#     targets = []

#     print("🔍 Validation started...")

#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device).float()

#             outputs = model(inputs)
#             probs = torch.sigmoid(outputs)

#             preds.extend(probs.cpu().numpy().flatten())
#             targets.extend(labels.cpu().numpy().flatten())

#     auc = roc_auc_score(targets, preds)

#     print("✅ Validation done")

#     return auc



# import time

# def train_model(model,train_loader,val_loader,device,pos_weight,epochs=5,lr=1e-4,save_path="best_model.pth"):
#     model = model.to(device)
#     criterion = nn.BCEWithLogitsLoss(
#         pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
#     )

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     train_losses = []
#     val_aucs = []

#     best_auc = 0.0

#     start_time = time.time()

#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0

#         print(f"\n🚀 Epoch [{epoch+1}/{epochs}]")

#         # ======================
#         # TRAIN
#         # ======================
#         for batch_idx, (inputs, labels) in enumerate(train_loader):
#             inputs = inputs.to(device)
#             labels = labels.to(device).float().unsqueeze(1)

#             optimizer.zero_grad()

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#             if batch_idx % 100 == 0:
#                 print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

#         avg_loss = running_loss / len(train_loader)
#         train_losses.append(avg_loss)

#         # ======================
#         # VALIDATION (AUC only)
#         # ======================
#         print("➡️ Running validation...")
#         val_auc = validate(model, val_loader, device)
#         val_aucs.append(val_auc)

#         print(f"📊 Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

#         # ======================
#         # SAVE BEST MODEL
#         # ======================
#         if val_auc > best_auc:
#             best_auc = val_auc
#             torch.save(model.state_dict(), save_path)
#             print("💾 Best model saved")

#     print(f"\n⏱ Training finished in {time.time() - start_time:.2f}s")
#     print(f"🔥 Best AUC: {best_auc:.4f}")

#     return train_losses, val_aucs

# class_counts = train_df["target"].value_counts()
# pos_weight = math.sqrt(class_counts[0] / class_counts[1])


# resnet_model = get_model("resnet50")

# print("\n====================")
# print("Training ResNet50")
# print("====================")

# resnet_losses, resnet_aucs = train_model(
#     model=resnet_model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     device=device,
#     pos_weight=pos_weight,
#     epochs=5,
#     lr=1e-4,
#     save_path="resnet50_best.pth"
# )

# efficient_model = get_model("efficientnet_b0")

# print("\n====================")
# print("Training EfficientNet-B0")
# print("====================")

# eff_losses, eff_aucs = train_model(
#     model=efficient_model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     device=device,
#     pos_weight=pos_weight,
#     epochs=5,
#     lr=5e-5,  # 🔥 փոքր LR EfficientNet-ի համար
#     save_path="efficientnet_best.pth"
# )

# print("\n📊 FINAL COMPARISON")

# print(f"ResNet50 Best AUC: {max(resnet_aucs):.4f}")
# print(f"EfficientNet Best AUC: {max(eff_aucs):.4f}")







import os
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import  roc_auc_score
from dataset import MelanomaDataset
from baseline_model import  get_model

from sklearn.model_selection import GroupKFold
from preprocess import prepare_resized_images
import albumentations as A
from albumentations.pytorch import ToTensorV2


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))

input_dir = DATASET_PATH / "jpeg" / "train"
output_dir = DATASET_PATH / "jpeg_224" / "train"

if not output_dir.exists() or len(os.listdir(output_dir)) == 0:
    prepare_resized_images(input_dir, output_dir)
else:
    print("Resized dataset already exists.")

train_path = output_dir
labels_path = DATASET_PATH / "train.csv"

df = pd.read_csv(labels_path)
df = df[["image_name", "patient_id", "target"]]

gkf = GroupKFold(n_splits=5)
df["fold"] = -1

for fold, (train_idx, val_idx) in enumerate(
    gkf.split(df, df["target"], groups=df["patient_id"])
):
    df.loc[val_idx, "fold"] = fold

train_df = df[df.fold != 0].reset_index(drop=True)
val_df = df[df.fold == 0].reset_index(drop=True)



train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])



train_dataset = MelanomaDataset(train_df, train_path, train_transform)
val_dataset = MelanomaDataset(val_df, train_path, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=2,pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

def validate(model, val_loader, device):
    model.eval()

    preds = []
    targets = []

    print("🔍 Validation started...")

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            outputs = model(inputs).squeeze(1) 
            probs = torch.sigmoid(outputs)

            preds.extend(probs.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())

    auc = roc_auc_score(targets, preds)

    print("✅ Validation done")

    return auc



import time

def train_model(model,train_loader,val_loader,device,epochs=5,lr=1e-4,save_path="best_model.pth"):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

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
        # VALIDATION (AUC only)
        # ======================
        print("➡️ Running validation...")
        val_auc = validate(model, val_loader, device)
        val_aucs.append(val_auc)

        print(f"📊 Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

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



resnet_model = get_model("resnet50")

print("\n====================")
print("Training ResNet50")
print("====================")

resnet_losses, resnet_aucs = train_model(
    model=resnet_model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=5,
    lr=1e-4,
    save_path="resnet50_best.pth"
)

del resnet_model
torch.cuda.empty_cache()


efficient_model = get_model("efficientnet_b0")

print("\n====================")
print("Training EfficientNet-B0")
print("====================")

eff_losses, eff_aucs = train_model(
    model=efficient_model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    epochs=5,
    lr=1e-4,
    save_path="efficientnet_best.pth"
)


print("\n📊 FINAL COMPARISON")

print(f"ResNet50 Best AUC: {max(resnet_aucs):.4f}")
print(f"EfficientNet Best AUC: {max(eff_aucs):.4f}")



