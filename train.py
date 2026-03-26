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
# # Paths
# # ======================
# DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))

# IMAGE_DIR = DATASET_PATH / "jpeg" / "train"
# CSV_PATH = DATASET_PATH / "train.csv"


# # ======================
# # Load CSV
# # ======================
# df = pd.read_csv(CSV_PATH)


# # ======================
# # Split (patient-wise)
# # ======================
# gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

# train_df = df.iloc[train_idx].reset_index(drop=True)
# val_df = df.iloc[val_idx].reset_index(drop=True)


# # ======================
# # Transforms
# # ======================
# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)), # Resize bigger just to see it clearly
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomRotation(15), # Rotate +/- 15 degrees
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.ToTensor(),
# ])

# val_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])


# # ======================
# # Dataset & Loader
# # ======================
# train_dataset = MelanomaDataset(train_df, IMAGE_DIR, train_transform)
# val_dataset = MelanomaDataset(val_df, IMAGE_DIR, val_transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32)


# # ======================
# # Model
# # ======================
# model = SimpleCNN().to(DEVICE)


# # ======================
# # Class weights (IMPORTANT)
# # ======================
# class_counts = train_df["target"].value_counts().sort_index()
# weights = 1.0 / class_counts
# weights = torch.tensor(weights.values, dtype=torch.float).to(DEVICE)

# criterion = nn.CrossEntropyLoss(weight=weights)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# # ======================
# # Training
# # ======================
# EPOCHS = 3

# for epoch in range(EPOCHS):
#     model.train()
#     train_loss = 0

#     for images, labels in train_loader:
#         images = images.to(DEVICE)
#         labels = labels.to(DEVICE)

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()

#     print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")


#     # ======================
#     # Validation
#     # ======================
#     model.eval()
#     preds, targets = [], []

#     with torch.no_grad():
#         for images, labels in val_loader:
#             images = images.to(DEVICE)
#             labels = labels.to(DEVICE)

#             outputs = model(images)
#             pred = torch.argmax(outputs, dim=1)

#             preds.extend(pred.cpu().numpy())
#             targets.extend(labels.cpu().numpy())

#     acc = accuracy_score(targets, preds)
#     f1 = f1_score(targets, preds)

#     print(f"Val Accuracy: {acc:.4f} | F1: {f1:.4f}")

# import os
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, WeightedRandomSampler
# from torchvision import transforms

# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     precision_score,
#     recall_score,
#     confusion_matrix,
#     classification_report,
# )

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
# # Paths
# # ======================
# DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))
# IMAGE_DIR = DATASET_PATH / "jpeg" / "train"
# CSV_PATH = DATASET_PATH / "train.csv"

# print("DATASET_PATH:", DATASET_PATH)
# print("IMAGE_DIR:", IMAGE_DIR)
# print("CSV_PATH:", CSV_PATH)


# # ======================
# # Config
# # ======================
# BATCH_SIZE = 64
# EPOCHS = 5
# LR = 1e-3
# NUM_WORKERS = 0
# IMAGE_SIZE = 128




# # ======================
# # Load CSV
# # ======================
# df = pd.read_csv(CSV_PATH)

# print("\nFull dataframe shape:", df.shape)
# print("Columns:", list(df.columns))
# print("\nFull target distribution:")
# print(df["target"].value_counts().sort_index())


# # ======================
# # Split (patient-wise)
# # ======================
# gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

# train_df = df.iloc[train_idx].reset_index(drop=True)
# val_df = df.iloc[val_idx].reset_index(drop=True)

# print("\nTrain shape:", train_df.shape)
# print("Val shape:", val_df.shape)

# print("\nTrain target distribution (ORIGINAL):")
# print(train_df["target"].value_counts().sort_index())

# print("\nVal target distribution:")
# print(val_df["target"].value_counts().sort_index())

# train_patients = set(train_df["patient_id"])
# val_patients = set(val_df["patient_id"])
# overlap = train_patients.intersection(val_patients)
# print("\nOverlapping patients:", len(overlap))


# # ======================
# # Transforms
# # ======================
# train_transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ToTensor(),
# ])

# val_transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
# ])


# # ======================
# # Dataset
# # ======================
# train_dataset = MelanomaDataset(train_df, IMAGE_DIR, transform=train_transform)
# val_dataset = MelanomaDataset(val_df, IMAGE_DIR, transform=val_transform)

# print("\nTrain dataset size:", len(train_dataset))
# print("Val dataset size:", len(val_dataset))


# # ======================
# # Imbalance handling
# # ======================
# # Class counts in original train_df
# class_counts = train_df["target"].value_counts().sort_index()
# print("\nClass counts in ORIGINAL train_df:")
# print(class_counts)

# # Class weights for loss
# class_weights_series = 1.0 / class_counts
# class_weights_tensor = torch.tensor(
#     class_weights_series.values,
#     dtype=torch.float32
# ).to(DEVICE)

# print("\nClass weights for loss:")
# print(class_weights_tensor)

# # Sample weights for sampler
# sample_weights = train_df["target"].map(class_weights_series).values
# sample_weights = torch.DoubleTensor(sample_weights)

# sampler = WeightedRandomSampler(
#     weights=sample_weights,
#     num_samples=len(sample_weights),
#     replacement=True
# )

# print("\nWeightedRandomSampler enabled.")


# # ======================
# # DataLoader
# # ======================
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     sampler=sampler,
#     num_workers=NUM_WORKERS,
#     pin_memory=True
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=NUM_WORKERS,
#     pin_memory=True
# )

# print("Train batches:", len(train_loader))
# print("Val batches:", len(val_loader))


# # ======================
# # Quick sampler check
# # ======================
# # We inspect a small sampled subset to verify that sampler is rebalancing
# sampled_labels_preview = []
# for i, (_, labels) in enumerate(train_loader):
#     sampled_labels_preview.extend(labels.numpy().tolist())
#     if len(sampled_labels_preview) >= 2000:
#         break

# sampled_labels_preview = np.array(sampled_labels_preview[:2000])
# unique_vals, unique_counts = np.unique(sampled_labels_preview, return_counts=True)

# print("\nSampler preview over first ~2000 sampled items:")
# for val, cnt in zip(unique_vals, unique_counts):
#     print(f"Class {val}: {cnt}")


# # ======================
# # Model
# # ======================
# model = SimpleCNN().to(DEVICE)


# # ======================
# # Loss + Optimizer
# # ======================
# criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# # ======================
# # Threshold search helper
# # ======================
# def find_best_threshold(y_true, y_prob):
#     thresholds = np.arange(0.10, 0.91, 0.05)

#     best_thr = 0.50
#     best_f1 = -1.0

#     for thr in thresholds:
#         y_pred = (y_prob >= thr).astype(int)
#         score = f1_score(y_true, y_pred, zero_division=0)

#         if score > best_f1:
#             best_f1 = score
#             best_thr = thr

#     return best_thr, best_f1


# # ======================
# # Training
# # ======================
# best_val_f1 = 0.0
# best_threshold = 0.50

# for epoch in range(EPOCHS):
#     print(f"\n{'=' * 50}")
#     print(f"Starting epoch {epoch + 1}/{EPOCHS}")
#     print(f"{'=' * 50}")

#     # ------------------
#     # Train
#     # ------------------
#     model.train()
#     train_loss = 0.0

#     for batch_idx, (images, labels) in enumerate(train_loader):
#         images = images.to(DEVICE, non_blocking=True)
#         labels = labels.to(DEVICE, non_blocking=True)

#         optimizer.zero_grad()

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()

#         if batch_idx % 100 == 0:
#             print(f"Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

#     avg_train_loss = train_loss / len(train_loader)
#     print(f"\nEpoch {epoch + 1} - Average Train Loss: {avg_train_loss:.4f}")

#     # ------------------
#     # Validation
#     # ------------------
#     model.eval()

#     val_loss = 0.0
#     all_probs = []
#     all_targets = []

#     with torch.no_grad():
#         for images, labels in val_loader:
#             images = images.to(DEVICE, non_blocking=True)
#             labels = labels.to(DEVICE, non_blocking=True)

#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()

#             probs = torch.softmax(outputs, dim=1)[:, 1]

#             all_probs.extend(probs.cpu().numpy())
#             all_targets.extend(labels.cpu().numpy())

#     all_probs = np.array(all_probs)
#     all_targets = np.array(all_targets)

#     avg_val_loss = val_loss / len(val_loader)

#     # Find best threshold on validation
#     thr, tuned_f1 = find_best_threshold(all_targets, all_probs)
#     preds = (all_probs >= thr).astype(int)

#     acc = accuracy_score(all_targets, preds)
#     precision = precision_score(all_targets, preds, zero_division=0)
#     recall = recall_score(all_targets, preds, zero_division=0)
#     f1 = f1_score(all_targets, preds, zero_division=0)
#     cm = confusion_matrix(all_targets, preds)

#     print(f"\nEpoch {epoch + 1} - Val Loss      : {avg_val_loss:.4f}")
#     print(f"Epoch {epoch + 1} - Best Threshold: {thr:.2f}")
#     print(f"Epoch {epoch + 1} - Val Accuracy  : {acc:.4f}")
#     print(f"Epoch {epoch + 1} - Val Precision : {precision:.4f}")
#     print(f"Epoch {epoch + 1} - Val Recall    : {recall:.4f}")
#     print(f"Epoch {epoch + 1} - Val F1        : {f1:.4f}")

#     print("\nConfusion Matrix:")
#     print(cm)

#     print("\nPrediction distribution on validation:")
#     pred_unique, pred_counts = np.unique(preds, return_counts=True)
#     for cls, cnt in zip(pred_unique, pred_counts):
#         print(f"Pred class {cls}: {cnt}")

#     print("\nTarget distribution on validation:")
#     target_unique, target_counts = np.unique(all_targets, return_counts=True)
#     for cls, cnt in zip(target_unique, target_counts):
#         print(f"True class {cls}: {cnt}")

#     print("\nClassification Report:")
#     print(classification_report(all_targets, preds, zero_division=0, digits=4))

#     if f1 > best_val_f1:
#         best_val_f1 = f1
#         best_threshold = thr
#         torch.save(
#             {
#                 "model_state_dict": model.state_dict(),
#                 "best_threshold": best_threshold,
#                 "best_val_f1": best_val_f1,
#                 "image_size": IMAGE_SIZE,
#             },
#             "model_best.pth"
#         )
#         print("Best model saved to model_best.pth")

# print("\nTraining finished.")
# print("Best Val F1:", best_val_f1)
# print("Best Threshold:", best_threshold)

# import os
# import copy
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# from dataset import MelanomaDataset
# from baseline_model import build_model


# # ======================
# # Config
# # ======================
# SEED = 42
# IMAGE_SIZE = 224
# BATCH_SIZE = 32
# EPOCHS = 5
# LR = 1e-4
# NUM_WORKERS = 0
# PATIENCE = 2

# DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))
# CSV_PATH = DATASET_PATH / "train.csv"
# IMAGE_DIR = DATASET_PATH / "jpeg" / "train"
# SAVE_PATH = Path("best_model.pth")


# # ======================
# # Reproducibility
# # ======================
# def set_seed(seed=42):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# set_seed(SEED)


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
# # Transforms
# # ======================
# train_transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     ),
# ])

# val_transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     ),
# ])


# # ======================
# # Load data
# # ======================
# df = pd.read_csv(CSV_PATH)

# print("Full dataframe shape:", df.shape)
# print(df[["image_name", "patient_id", "target"]].head())


# # ======================
# # Patient-wise split
# # ======================
# gss = GroupShuffleSplit(
#     n_splits=1,
#     test_size=0.2,
#     random_state=SEED
# )

# train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

# train_df = df.iloc[train_idx].reset_index(drop=True)
# val_df = df.iloc[val_idx].reset_index(drop=True)

# print("Train shape:", train_df.shape)
# print("Val shape:", val_df.shape)

# overlap = set(train_df["patient_id"]).intersection(set(val_df["patient_id"]))
# print("Overlapping patients:", len(overlap))


# # ======================
# # Datasets
# # ======================
# train_dataset = MelanomaDataset(
#     df=train_df,
#     image_dir=IMAGE_DIR,
#     transform=train_transform
# )

# val_dataset = MelanomaDataset(
#     df=val_df,
#     image_dir=IMAGE_DIR,
#     transform=val_transform
# )


# # ======================
# # DataLoaders
# # ======================
# pin_memory = DEVICE.type == "cuda"

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=NUM_WORKERS,
#     pin_memory=pin_memory
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=NUM_WORKERS,
#     pin_memory=pin_memory
# )


# # ======================
# # Class weights
# # ======================
# class_counts = train_df["target"].value_counts().sort_index()
# print("Class counts:")
# print(class_counts)

# num_neg = class_counts.get(0, 1)
# num_pos = class_counts.get(1, 1)

# class_weights = torch.tensor(
#     [1.0, num_neg / num_pos],
#     dtype=torch.float32,
#     device=DEVICE
# )

# print("Class weights:", class_weights)


# # ======================
# # Model / Loss / Optimizer
# # ======================
# model = build_model(num_classes=2).to(DEVICE)

# criterion = nn.CrossEntropyLoss(weight=class_weights)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# use_amp = DEVICE.type == "cuda"
# scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


# # ======================
# # Train function
# # ======================
# def train_one_epoch(model, loader, criterion, optimizer, device):
#     model.train()

#     running_loss = 0.0
#     all_preds = []
#     all_labels = []

#     for batch_idx, (images, labels) in enumerate(loader):
#         images = images.to(device)
#         labels = labels.to(device)

#         optimizer.zero_grad()

#         with torch.cuda.amp.autocast(enabled=use_amp):
#             outputs = model(images)
#             loss = criterion(outputs, labels)

#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         running_loss += loss.item() * images.size(0)

#         preds = torch.argmax(outputs, dim=1)
#         all_preds.extend(preds.detach().cpu().numpy())
#         all_labels.extend(labels.detach().cpu().numpy())

#         if batch_idx % 100 == 0:
#             print(f"Train batch {batch_idx}/{len(loader)}")

#     epoch_loss = running_loss / len(loader.dataset)
#     epoch_acc = accuracy_score(all_labels, all_preds)
#     epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)

#     return epoch_loss, epoch_acc, epoch_f1


# # ======================
# # Validation function
# # ======================
# @torch.no_grad()
# def validate_one_epoch(model, loader, criterion, device):
#     model.eval()

#     running_loss = 0.0
#     all_preds = []
#     all_labels = []

#     for images, labels in loader:
#         images = images.to(device)
#         labels = labels.to(device)

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         running_loss += loss.item() * images.size(0)

#         preds = torch.argmax(outputs, dim=1)
#         all_preds.extend(preds.detach().cpu().numpy())
#         all_labels.extend(labels.detach().cpu().numpy())

#     epoch_loss = running_loss / len(loader.dataset)
#     epoch_acc = accuracy_score(all_labels, all_preds)
#     epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
#     epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
#     epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
#     epoch_cm = confusion_matrix(all_labels, all_preds)

#     return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall, epoch_cm


# # ======================
# # Training loop
# # ======================
# best_f1 = -1.0
# best_model_wts = copy.deepcopy(model.state_dict())
# early_stop_counter = 0

# for epoch in range(EPOCHS):
#     print(f"\nEpoch {epoch + 1}/{EPOCHS}")

#     train_loss, train_acc, train_f1 = train_one_epoch(
#         model, train_loader, criterion, optimizer, DEVICE
#     )

#     val_loss, val_acc, val_f1, val_precision, val_recall, val_cm = validate_one_epoch(
#         model, val_loader, criterion, DEVICE
#     )

#     print(f"\nResults for epoch {epoch + 1}")
#     print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
#     print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   F1: {val_f1:.4f}")
#     print(f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f}")
#     print("Val Confusion Matrix:")
#     print(val_cm)

#     if val_f1 > best_f1:
#         best_f1 = val_f1
#         best_model_wts = copy.deepcopy(model.state_dict())
#         torch.save(model.state_dict(), SAVE_PATH)
#         print(f"Best model saved to: {SAVE_PATH}")
#         early_stop_counter = 0
#     else:
#         early_stop_counter += 1
#         print(f"No improvement. Early stop counter: {early_stop_counter}/{PATIENCE}")

#     if early_stop_counter >= PATIENCE:
#         print("Early stopping triggered.")
#         break


# # ======================
# # Final
# # ======================
# model.load_state_dict(best_model_wts)
# print(f"\nTraining finished. Best Val F1: {best_f1:.4f}")

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from dataset import MelanomaDataset
from baseline_model import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"

# paths
CSV_PATH = "dataset/train.csv"
IMG_DIR = "dataset/jpeg/train"

# transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# load data
df = pd.read_csv(CSV_PATH)

# imbalance fix
num_pos = df['target'].sum()
num_neg = len(df) - num_pos
pos_weight = torch.tensor([num_neg / num_pos]).to(device)

# split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['target'])

train_dataset = MelanomaDataset(train_df, IMG_DIR, train_transform)
val_dataset = MelanomaDataset(val_df, IMG_DIR, val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# model
model = get_model().to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_auc = 0

for epoch in range(5):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    # validation
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            preds.extend(probs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    auc = roc_auc_score(targets, preds)
    print(f"Epoch {epoch+1} AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), "baseline_best.pth")
        print("✅ saved best model")