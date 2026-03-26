# print("SCRIPT STARTED")

# import os
# from pathlib import Path
# import pandas as pd
# import numpy as np

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import transforms

# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.metrics import accuracy_score, f1_score

# from dataset import MelanomaDataset
# from baseline_model import SimpleCNN

# print(torch.backends.mps.is_available())

# if torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
# else:
#     DEVICE = torch.device("cpu")

# # ======================
# # Config
# # ======================

# DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))

# IMAGE_DIR = DATASET_PATH / "jpeg" / "train"
# CSV_PATH = DATASET_PATH / "train.csv"

# BATCH_SIZE = 32
# EPOCHS = 3
# LR = 1e-3
# PATIENCE = 3



# # ======================
# # Load data
# # ======================

# df = pd.read_csv(CSV_PATH)
# df = df[["image_name", "patient_id", "target"]]


# # ======================
# # Patient-wise split
# # ======================

# gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

# train_df = df.iloc[train_idx].reset_index(drop=True)
# val_df = df.iloc[val_idx].reset_index(drop=True)

# train_df = train_df[["image_name", "target"]]
# val_df = val_df[["image_name", "target"]]


# # ======================
# # Class weights
# # ======================

# class_counts = train_df["target"].value_counts().sort_index().values

# class_weights = torch.tensor(
#     [len(train_df) / class_counts[0], len(train_df) / class_counts[1]],
#     dtype=torch.float32
# ).to(DEVICE)

# print("Class counts:", class_counts)
# print("Class weights:", class_weights)


# # ======================
# # Transforms
# # ======================

# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1),
#     transforms.ToTensor()
# ])

# val_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])


# # ======================
# # Dataset & Loader
# # ======================
# print("Creating datasets and dataloaders...")
# train_dataset = MelanomaDataset(train_df, IMAGE_DIR, transform=train_transform)
# val_dataset = MelanomaDataset(val_df, IMAGE_DIR, transform=val_transform)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# # ======================
# # Model
# # ======================
# print("Initializing model...")
# model = SimpleCNN().to(DEVICE)

# criterion = nn.CrossEntropyLoss(weight=class_weights)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# # ======================
# # Train function
# # ======================
# print("Defining training function...")
# def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience):
#     best_val_loss = float("inf")
#     early_stop_counter = 0

#     for epoch in range(epochs):

#         # train
#         model.train()
#         train_loss = 0
#         train_preds = []
#         train_labels = []

#         for images, labels in train_loader:
#             images = images.to(DEVICE)
#             labels = labels.to(DEVICE)

#             optimizer.zero_grad()

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()

#             preds = torch.argmax(outputs, dim=1)
#             train_preds.extend(preds.cpu().numpy())
#             train_labels.extend(labels.cpu().numpy())
#         print("Finished training loop for epoch", epoch+1)
#         train_loss = train_loss / len(train_loader)
#         train_acc = accuracy_score(train_labels, train_preds)
#         train_f1 = f1_score(train_labels, train_preds)


#         # validation
#         model.eval()
#         val_loss = 0
#         val_preds = []
#         val_labels = []

#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images = images.to(DEVICE)
#                 labels = labels.to(DEVICE)

#                 outputs = model(images)
#                 loss = criterion(outputs, labels)

#                 val_loss += loss.item()

#                 preds = torch.argmax(outputs, dim=1)
#                 val_preds.extend(preds.cpu().numpy())
#                 val_labels.extend(labels.cpu().numpy())

#         val_loss = val_loss / len(val_loader)
#         val_acc = accuracy_score(val_labels, val_preds)
#         val_f1 = f1_score(val_labels, val_preds)

#         print(f"Epoch {epoch+1}/{epochs}")
#         print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
#         print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")


#         # early stopping
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             early_stop_counter = 0

#             torch.save(model.state_dict(), "baseline_best.pth")
#             print("Best model saved.\n")
#         else:
#             early_stop_counter += 1
#             print(f"No improvement. Early stop counter: {early_stop_counter}/{patience}\n")

#         if early_stop_counter >= patience:
#             print("Early stopping triggered.")
#             break


# # ======================
# # Run training
# # ======================

# if __name__ == "__main__":
#     train_model(
#         model,
#         train_loader,
#         val_loader,
#         criterion,
#         optimizer,
#         EPOCHS,
#         PATIENCE
#     )


# import os
# from pathlib import Path
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import transforms

# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.metrics import accuracy_score, f1_score

# from dataset import MelanomaDataset
# from baseline_model import SimpleCNN


# print("SCRIPT STARTED")


# # ======================
# # Device
# # ======================

# if torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
# else:
#     DEVICE = torch.device("cpu")

# print("Using device:", DEVICE)


# # ======================
# # Config
# # ======================

# DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))

# IMAGE_DIR = DATASET_PATH / "jpeg" / "train"
# CSV_PATH = DATASET_PATH / "train.csv"

# BATCH_SIZE = 16
# EPOCHS = 3
# LR = 1e-3
# PATIENCE = 3

# print("CSV_PATH:", CSV_PATH)
# print("IMAGE_DIR:", IMAGE_DIR)


# # ======================
# # Load data
# # ======================

# df = pd.read_csv(CSV_PATH)
# print("CSV loaded:", df.shape)

# df = df[["image_name", "patient_id", "target"]]
# print("Selected columns:", df.shape)


# # ======================
# # Patient-wise split
# # ======================

# gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

# train_df = df.iloc[train_idx].reset_index(drop=True)
# val_df = df.iloc[val_idx].reset_index(drop=True)

# print("Split done")
# print("train_df:", train_df.shape)
# print("val_df:", val_df.shape)

# train_df = train_df[["image_name", "target"]]
# val_df = val_df[["image_name", "target"]]


# # ======================
# # Class weights
# # ======================

# class_counts = train_df["target"].value_counts().sort_index().values

# class_weights = torch.tensor(
#     [len(train_df) / class_counts[0], len(train_df) / class_counts[1]],
#     dtype=torch.float32
# ).to(DEVICE)

# print("Class counts:", class_counts)
# print("Class weights:", class_weights)


# # ======================
# # Transforms
# # ======================

# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1),
#     transforms.ToTensor()
# ])

# val_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# print("Transforms ready")


# # ======================
# # Dataset & Loader
# # ======================

# train_dataset = MelanomaDataset(train_df, IMAGE_DIR, transform=train_transform)
# val_dataset = MelanomaDataset(val_df, IMAGE_DIR, transform=val_transform)

# print("Datasets created")
# print("Train dataset len:", len(train_dataset))
# print("Val dataset len:", len(val_dataset))

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# print("Dataloaders created")
# print("Train loader len:", len(train_loader))
# print("Val loader len:", len(val_loader))


# # ======================
# # Model
# # ======================

# model = SimpleCNN().to(DEVICE)

# criterion = nn.CrossEntropyLoss(weight=class_weights)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# print("Model initialized")


# # ======================
# # Train function
# # ======================

# def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience):
#     best_val_loss = float("inf")
#     early_stop_counter = 0

#     train_losses = []
#     val_losses = []
#     train_accs = []
#     val_accs = []
#     train_f1s = []
#     val_f1s = []

#     print("Entered train_model")

#     for epoch in range(epochs):
#         print(f"\nStarting epoch {epoch+1}/{epochs}")

#         # ======================
#         # Train
#         # ======================
#         model.train()
#         train_loss = 0
#         train_preds = []
#         train_labels = []

#         for i, (images, labels) in enumerate(train_loader):
#             if i == 0:
#                 print("First train batch loaded")

#             images = images.to(DEVICE)
#             labels = labels.to(DEVICE)

#             optimizer.zero_grad()

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()

#             preds = torch.argmax(outputs, dim=1)
#             train_preds.extend(preds.cpu().numpy())
#             train_labels.extend(labels.cpu().numpy())

#             if i % 100 == 0:
#                 print(f"Epoch {epoch+1} - Train Batch {i}/{len(train_loader)}")

#         train_loss = train_loss / len(train_loader)
#         train_acc = accuracy_score(train_labels, train_preds)
#         train_f1 = f1_score(train_labels, train_preds)

#         # ======================
#         # Validation
#         # ======================
#         model.eval()
#         val_loss = 0
#         val_preds = []
#         val_labels = []

#         with torch.no_grad():
#             for i, (images, labels) in enumerate(val_loader):
#                 if i == 0:
#                     print("First val batch loaded")

#                 images = images.to(DEVICE)
#                 labels = labels.to(DEVICE)

#                 outputs = model(images)
#                 loss = criterion(outputs, labels)

#                 val_loss += loss.item()

#                 preds = torch.argmax(outputs, dim=1)
#                 val_preds.extend(preds.cpu().numpy())
#                 val_labels.extend(labels.cpu().numpy())

#         val_loss = val_loss / len(val_loader)
#         val_acc = accuracy_score(val_labels, val_preds)
#         val_f1 = f1_score(val_labels, val_preds)

#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#         train_accs.append(train_acc)
#         val_accs.append(val_acc)
#         train_f1s.append(train_f1)
#         val_f1s.append(val_f1)

#         print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
#         print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")

#         # ======================
#         # Early stopping
#         # ======================
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             early_stop_counter = 0

#             torch.save(model.state_dict(), "baseline_best.pth")
#             print("Best model saved.")
#         else:
#             early_stop_counter += 1
#             print(f"No improvement. Early stop counter: {early_stop_counter}/{patience}")

#         if early_stop_counter >= patience:
#             print("Early stopping triggered.")
#             break

#     return train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s


# # ======================
# # Run training
# # ======================

# if __name__ == "__main__":
#     train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s = train_model(
#         model,
#         train_loader,
#         val_loader,
#         criterion,
#         optimizer,
#         EPOCHS,
#         PATIENCE
#     )

#     print("\nTraining finished.")
#     print("Train losses:", train_losses)
#     print("Val losses:", val_losses)


# import os
# from pathlib import Path

# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import transforms

# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.metrics import accuracy_score, f1_score

# from dataset import MelanomaDataset
# from baseline_model import SimpleCNN


# print("SCRIPT STARTED")


# # ======================
# # Device
# # ======================

# if torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     DEVICE = torch.device("mps")
# else:
#     DEVICE = torch.device("cpu")

# print("Using device:", DEVICE)

# if DEVICE.type == "cuda":
#     print("GPU:", torch.cuda.get_device_name(0))


# # ======================
# # Config
# # ======================

# DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))

# IMAGE_DIR = DATASET_PATH / "jpeg" / "train"
# CSV_PATH = DATASET_PATH / "train.csv"

# BATCH_SIZE = 32
# EPOCHS = 3
# LR = 1e-3
# PATIENCE = 3
# NUM_WORKERS = 2 if DEVICE.type == "cuda" else 0

# print("CSV_PATH:", CSV_PATH)
# print("IMAGE_DIR:", IMAGE_DIR)


# # ======================
# # Safety checks
# # ======================

# if not CSV_PATH.exists():
#     raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

# if not IMAGE_DIR.exists():
#     raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")


# # ======================
# # Load data
# # ======================

# df = pd.read_csv(CSV_PATH)
# print("CSV loaded:", df.shape)

# df = df[["image_name", "patient_id", "target"]].dropna().reset_index(drop=True)
# df["target"] = df["target"].astype(int)

# print("Selected columns:", df.shape)


# # ======================
# # Patient-wise split
# # ======================

# gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

# train_df = df.iloc[train_idx].reset_index(drop=True)
# val_df = df.iloc[val_idx].reset_index(drop=True)

# print("Split done")
# print("train_df:", train_df.shape)
# print("val_df:", val_df.shape)

# overlap = set(train_df["patient_id"]).intersection(set(val_df["patient_id"]))
# print("Overlapping patients:", len(overlap))

# train_df = train_df[["image_name", "target"]]
# val_df = val_df[["image_name", "target"]]


# # ======================
# # Class weights
# # ======================

# class_counts = train_df["target"].value_counts().sort_index()

# if len(class_counts) < 2:
#     raise ValueError("Training split must contain both classes (0 and 1).")

# class_weights = torch.tensor(
#     [
#         len(train_df) / class_counts[0],
#         len(train_df) / class_counts[1],
#     ],
#     dtype=torch.float32
# ).to(DEVICE)

# print("Class counts:")
# print(class_counts)
# print("Class weights:", class_weights)


# # ======================
# # Transforms
# # ======================

# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1),
#     transforms.ToTensor(),
# ])

# val_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# print("Transforms ready")


# # ======================
# # Dataset & Loader
# # ======================

# train_dataset = MelanomaDataset(train_df, IMAGE_DIR, transform=train_transform)
# val_dataset = MelanomaDataset(val_df, IMAGE_DIR, transform=val_transform)

# print("Datasets created")
# print("Train dataset len:", len(train_dataset))
# print("Val dataset len:", len(val_dataset))

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=NUM_WORKERS,
#     pin_memory=(DEVICE.type == "cuda"),
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=NUM_WORKERS,
#     pin_memory=(DEVICE.type == "cuda"),
# )

# print("Dataloaders created")
# print("Train loader len:", len(train_loader))
# print("Val loader len:", len(val_loader))


# # ======================
# # Model
# # ======================

# model = SimpleCNN().to(DEVICE)

# criterion = nn.CrossEntropyLoss(weight=class_weights)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# print("Model initialized")


# # ======================
# # Train function
# # ======================

# def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience):
#     best_val_loss = float("inf")
#     early_stop_counter = 0

#     train_losses = []
#     val_losses = []
#     train_accs = []
#     val_accs = []
#     train_f1s = []
#     val_f1s = []

#     print("Entered train_model")

#     for epoch in range(epochs):
#         print(f"\nStarting epoch {epoch + 1}/{epochs}")

#         # ======================
#         # Train
#         # ======================
#         model.train()
#         train_loss = 0.0
#         train_preds = []
#         train_labels = []

#         for i, (images, labels) in enumerate(train_loader):
#             if i == 0:
#                 print("First train batch loaded")

#             images = images.to(DEVICE, non_blocking=True)
#             labels = labels.to(DEVICE, non_blocking=True).long()

#             optimizer.zero_grad()

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()

#             preds = torch.argmax(outputs, dim=1)
#             train_preds.extend(preds.detach().cpu().numpy())
#             train_labels.extend(labels.detach().cpu().numpy())

#             if i % 100 == 0:
#                 print(f"Epoch {epoch + 1} - Train Batch {i}/{len(train_loader)}")

#         train_loss = train_loss / len(train_loader)
#         train_acc = accuracy_score(train_labels, train_preds)
#         train_f1 = f1_score(train_labels, train_preds, zero_division=0)

#         # ======================
#         # Validation
#         # ======================
#         model.eval()
#         val_loss = 0.0
#         val_preds = []
#         val_labels = []

#         with torch.no_grad():
#             for i, (images, labels) in enumerate(val_loader):
#                 if i == 0:
#                     print("First val batch loaded")

#                 images = images.to(DEVICE, non_blocking=True)
#                 labels = labels.to(DEVICE, non_blocking=True).long()

#                 outputs = model(images)
#                 loss = criterion(outputs, labels)

#                 val_loss += loss.item()

#                 preds = torch.argmax(outputs, dim=1)
#                 val_preds.extend(preds.detach().cpu().numpy())
#                 val_labels.extend(labels.detach().cpu().numpy())

#         val_loss = val_loss / len(val_loader)
#         val_acc = accuracy_score(val_labels, val_preds)
#         val_f1 = f1_score(val_labels, val_preds, zero_division=0)

#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#         train_accs.append(train_acc)
#         val_accs.append(val_acc)
#         train_f1s.append(train_f1)
#         val_f1s.append(val_f1)

#         print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
#         print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")

#         # ======================
#         # Early stopping
#         # ======================
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             early_stop_counter = 0

#             torch.save(model.state_dict(), "baseline_best.pth")
#             print("Best model saved.")
#         else:
#             early_stop_counter += 1
#             print(f"No improvement. Early stop counter: {early_stop_counter}/{patience}")

#         if early_stop_counter >= patience:
#             print("Early stopping triggered.")
#             break

#     return train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s


# # ======================
# # Run training
# # ======================

# if __name__ == "__main__":
#     train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s = train_model(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         criterion=criterion,
#         optimizer=optimizer,
#         epochs=EPOCHS,
#         patience=PATIENCE,
#     )

#     print("\nTraining finished.")
#     print("Train losses:", train_losses)
#     print("Val losses:", val_losses)
#     print("Train accs:", train_accs)
#     print("Val accs:", val_accs)
#     print("Train f1s:", train_f1s)
#     print("Val f1s:", val_f1s)



import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

from dataset import MelanomaDataset
from baseline_model import SimpleCNN


# ======================
# Device
# ======================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ======================
# Paths
# ======================
DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))

IMAGE_DIR = DATASET_PATH / "jpeg" / "train"
CSV_PATH = DATASET_PATH / "train.csv"


# ======================
# Load CSV
# ======================
df = pd.read_csv(CSV_PATH)


# ======================
# Split (patient-wise)
# ======================
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)


# ======================
# Transforms
# ======================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ======================
# Dataset & Loader
# ======================
train_dataset = MelanomaDataset(train_df, IMAGE_DIR, train_transform)
val_dataset = MelanomaDataset(val_df, IMAGE_DIR, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# ======================
# Model
# ======================
model = SimpleCNN().to(DEVICE)


# ======================
# Class weights (IMPORTANT)
# ======================
class_counts = train_df["target"].value_counts().sort_index()
weights = 1.0 / class_counts
weights = torch.tensor(weights.values, dtype=torch.float).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ======================
# Training
# ======================
EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")


    # ======================
    # Validation
    # ======================
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)

            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)

    print(f"Val Accuracy: {acc:.4f} | F1: {f1:.4f}")