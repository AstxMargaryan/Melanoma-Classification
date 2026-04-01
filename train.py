# import os
# from collections import Counter

# import pandas as pd
# import torch
# import torch.nn as nn
# from pathlib import Path
# from PIL import Image
# from tqdm import tqdm
# from torch.utils.data import DataLoader, WeightedRandomSampler
# from torchvision import transforms
# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# from dataset import MelanomaDataset
# from baseline_model import BaselineCNN, get_resnet

# # Device
# # ======================
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

# print("Using device:", device)


# def prepare_resized_images(input_dir, output_dir, size=(224, 224)):
#     input_dir = Path(input_dir)
#     output_dir = Path(output_dir)

#     if not input_dir.exists():
#         raise FileNotFoundError(f"Input folder not found: {input_dir}")

#     output_dir.mkdir(parents=True, exist_ok=True)

#     image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#     print(f"Total images found: {len(image_files)}")

#     for img_name in tqdm(image_files):
#         input_path = input_dir / img_name
#         output_path = output_dir / img_name

#         if output_path.exists():
#             continue

#         try:
#             img = Image.open(input_path).convert("RGB")
#             img = img.resize(size)
#             img.save(output_path)
#         except Exception as e:
#             print(f"❌ Error with {img_name}: {e}")

#     print("✅ Resized dataset ready.")



# DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))

# input_dir = DATASET_PATH / "jpeg" / "train"
# output_dir = DATASET_PATH / "jpeg_224" / "train"

# train_path = DATASET_PATH / "jpeg_224" / "train"
# test_path = DATASET_PATH / "jpeg" / "test"

# labels_path = DATASET_PATH / "train.csv"
# without_labels_path = DATASET_PATH / "test.csv"


# if not output_dir.exists() or len(os.listdir(output_dir)) == 0:
#     prepare_resized_images(input_dir, output_dir, size=(224, 224))
# else:
#     print("✅ Resized dataset already exists.")

# # # Read dataframe

# df = pd.read_csv(labels_path)
# test_df = pd.read_csv(without_labels_path)
# df = df[["image_name", "patient_id", "target"]]


# # ======================
# # Patient-wise split
# # ======================
# gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

# train_df = df.iloc[train_idx].reset_index(drop=True)
# val_df = df.iloc[val_idx].reset_index(drop=True)

# print("Train shape:", train_df.shape)
# print("Val shape:", val_df.shape)

# overlap = set(train_df["patient_id"]).intersection(set(val_df["patient_id"]))
# print("Overlapping patients:", len(overlap))



# # ======================
# # Transforms
# # ======================


# train_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(20),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# val_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])



# # ======================
# # Datasets
# # ======================
# cnn_train_dataset = MelanomaDataset(train_df, train_path, train_transform)
# cnn_val_dataset = MelanomaDataset(val_df, train_path, val_transform)

# resnet_train_dataset = MelanomaDataset(train_df, train_path, train_transform)
# resnet_val_dataset = MelanomaDataset(val_df, train_path, val_transform)


# def make_sampler(df):
#     pos_weight = (df["target"] == 0).sum() / (df["target"] == 1).sum()
#     weights = df["target"].map({0: 1.0, 1: pos_weight}).values
#     return WeightedRandomSampler(
#         weights=torch.tensor(weights, dtype=torch.float32),
#         num_samples=len(weights),
#         replacement=True
#     )

# cnn_sampler = make_sampler(train_df)
# resnet_sampler = make_sampler(train_df)


# # ======================
# # DataLoaders
# # ======================
 
# cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=32, sampler=cnn_sampler)
# cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=32, shuffle=False)

# resnet_train_loader = DataLoader(resnet_train_dataset, batch_size=32, sampler=resnet_sampler)
# resnet_val_loader = DataLoader(resnet_val_dataset, batch_size=32, shuffle=False)


# # ======================
# # Check sampler
# # ======================
# print("\n===== CHECK SAMPLER =====")
# batch_counter = Counter()

# for i, (_, labels) in enumerate(cnn_train_loader):
#     batch_counter.update(labels.tolist())
#     if i == 100:
#         break

# total = batch_counter[0.0] + batch_counter[1.0]

# print("Sampled counts:", batch_counter)
# print("Class 0 %:", 100 * batch_counter[0.0] / total)
# print("Class 1 %:", 100 * batch_counter[1.0] / total)



# # ======================
# # Validation
# # ======================
# def validate(model, val_loader, device, criterion):
#     model.eval()
#     val_loss = 0.0

#     all_labels = []
#     all_preds = []
#     all_probs = []

#     print("🔍 Validation started...")

#     with torch.no_grad():
#         for batch_idx, (images, labels) in enumerate(val_loader):
#             images = images.to(device, non_blocking=True)
#             labels = labels.to(device).float().unsqueeze(1)

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             val_loss += loss.item()

#             probs = torch.sigmoid(outputs)
#             preds = (probs >= 0.5).float()

#             all_labels.extend(labels.cpu().numpy().flatten())
#             all_probs.extend(probs.cpu().numpy().flatten()) 
#             all_preds.extend(preds.cpu().numpy().flatten())

#             if batch_idx % 100 == 0:
#                 print(f"Val Batch {batch_idx}/{len(val_loader)} | Loss: {loss.item():.4f}")

#     val_loss = val_loss / len(val_loader)

#     val_acc = accuracy_score(all_labels, all_preds)
#     val_f1 = f1_score(all_labels, all_preds, zero_division=0)
#     val_recall = recall_score(all_labels, all_preds, zero_division=0)
#     val_precision = precision_score(all_labels, all_preds, zero_division=0)

#     print("✅ Validation done")

#     return val_loss, val_acc, val_f1, val_recall, val_precision
# # ======================
# # Training
# # ======================
# def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001, save_path="model_best.pth"):
#     model = model.to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.BCEWithLogitsLoss()

#     train_losses = []
#     val_losses = []
#     val_accuracies = []
#     val_f1s = []
#     val_recalls = []
#     val_precisions = []

#     best_val_f1 = 0.0

#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0

#         print(f"\n🚀 Starting Epoch {epoch+1}/{epochs}")

#         for batch_idx, (images, labels) in enumerate(train_loader):
#             images = images.to(device)
#             labels = labels.to(device).float().unsqueeze(1)

#             optimizer.zero_grad()

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#             if batch_idx % 100 == 0:
#                 print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

#         epoch_train_loss = running_loss / len(train_loader)

#         print("➡️ Running validation...")
#         epoch_val_loss, val_acc, val_f1, val_recall, val_precision = validate(
#             model, val_loader, device, criterion
#         )

#         train_losses.append(epoch_train_loss)
#         val_losses.append(epoch_val_loss)
#         val_accuracies.append(val_acc)
#         val_f1s.append(val_f1)
#         val_recalls.append(val_recall)
#         val_precisions.append(val_precision)

#         if val_f1 > best_val_f1:
#             best_val_f1 = val_f1
#             torch.save(model.state_dict(), save_path)
#             print(f"💾 Best model saved to {save_path} | Val F1: {val_f1:.4f}")

#         print(
#             f"📊 Epoch {epoch+1}/{epochs} DONE | "
#             f"Train Loss: {epoch_train_loss:.4f} | "
#             f"Val Loss: {epoch_val_loss:.4f} | "
#             f"Val Acc: {val_acc:.4f} | "
#             f"Val F1: {val_f1:.4f} | "
#             f"Val Recall: {val_recall:.4f} | "
#             f"Val Precision: {val_precision:.4f}"
#         )

#     return {
#         "model": model,
#         "train_losses": train_losses,
#         "val_losses": val_losses,
#         "val_accuracies": val_accuracies,
#         "val_f1s": val_f1s,
#         "val_recalls": val_recalls,
#         "val_precisions": val_precisions,
#     }


# # ======================
# # Train CNN
# # ======================
# cnn_model = BaselineCNN()

# cnn_results = train_model(
#     model=cnn_model,
#     train_loader=cnn_train_loader,
#     val_loader=cnn_val_loader,
#     device=device,
#     epochs=10,
#     lr=0.001,
#     save_path="cnn_model_best.pth"
# )



# resnet_model = get_resnet()

# resnet_results = train_model(
#     model=resnet_model,
#     train_loader=resnet_train_loader,
#     val_loader=resnet_val_loader,
#     device=device,
#     epochs=10,
#     lr=0.0001,        # lower LR for pretrained model
#     save_path="resnet_model_best.pth"
# )

# ======================
# Train ResNet18
# ======================
# resnet_model = get_resnet()

# resnet_results = train_model(
#     model=resnet_model,
#     train_loader=resnet_train_loader,
#     val_loader=resnet_val_loader,
#     device=device,
#     pos_weight=pos_weight,
#     epochs=10,
#     lr=0.0001,
#     save_path="resnet_model_best.pth"
# )
# ======================
# Imbalance
# ======================
# class_counts = train_df["target"].value_counts().sort_index()
# num_neg = class_counts[0]
# num_pos = class_counts[1]
# pos_weight = num_neg / num_pos

# print("Imbalance ratio:", pos_weight)

# # ======================
# # Sampler
# # ======================
# sample_weights = train_df["target"].map({
#     0: 1.0,
#     1: pos_weight
# }).values

# sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

# sampler = WeightedRandomSampler(
#     weights=sample_weights,
#     num_samples=len(sample_weights),
#     replacement=True
# )


# cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=32,sampler=sampler)
# cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=32, shuffle=False)

# resnet_train_loader = DataLoader(resnet_train_dataset, batch_size=32, sampler=sampler)
# resnet_val_loader = DataLoader(resnet_val_dataset, batch_size=32, shuffle=False)


# import math
# import os
# from collections import Counter

# import pandas as pd
# import torch
# import torch.nn as nn
# from pathlib import Path
# from PIL import Image
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# from dataset import MelanomaDataset
# from baseline_model import BaselineCNN, get_resnet

# # Device
# # ======================
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

# print("Using device:", device)


# def prepare_resized_images(input_dir, output_dir, size=(224, 224)):
#     input_dir = Path(input_dir)
#     output_dir = Path(output_dir)

#     if not input_dir.exists():
#         raise FileNotFoundError(f"Input folder not found: {input_dir}")

#     output_dir.mkdir(parents=True, exist_ok=True)

#     image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#     print(f"Total images found: {len(image_files)}")

#     for img_name in tqdm(image_files):
#         input_path = input_dir / img_name
#         output_path = output_dir / img_name

#         if output_path.exists():
#             continue

#         try:
#             img = Image.open(input_path).convert("RGB")
#             img = img.resize(size)
#             img.save(output_path)
#         except Exception as e:
#             print(f"❌ Error with {img_name}: {e}")

#     print("✅ Resized dataset ready.")



# DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))

# input_dir = DATASET_PATH / "jpeg" / "train"
# output_dir = DATASET_PATH / "jpeg_224" / "train"

# train_path = DATASET_PATH / "jpeg_224" / "train"
# test_path = DATASET_PATH / "jpeg" / "test"

# labels_path = DATASET_PATH / "train.csv"
# without_labels_path = DATASET_PATH / "test.csv"


# if not output_dir.exists() or len(os.listdir(output_dir)) == 0:
#     prepare_resized_images(input_dir, output_dir, size=(224, 224))
# else:
#     print("✅ Resized dataset already exists.")

# # # Read dataframe

# df = pd.read_csv(labels_path)
# test_df = pd.read_csv(without_labels_path)
# df = df[["image_name", "patient_id", "target"]]


# # ======================
# # Patient-wise split
# # ======================
# gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

# train_df = df.iloc[train_idx].reset_index(drop=True)
# val_df = df.iloc[val_idx].reset_index(drop=True)

# print("Train shape:", train_df.shape)
# print("Val shape:", val_df.shape)

# overlap = set(train_df["patient_id"]).intersection(set(val_df["patient_id"]))
# print("Overlapping patients:", len(overlap))



# # ======================
# # Transforms
# # ======================


# train_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(20),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# val_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])



# # ======================
# # Datasets
# # ======================
# cnn_train_dataset = MelanomaDataset(train_df, train_path, train_transform)
# cnn_val_dataset = MelanomaDataset(val_df, train_path, val_transform)

# resnet_train_dataset = MelanomaDataset(train_df, train_path, train_transform)
# resnet_val_dataset = MelanomaDataset(val_df, train_path, val_transform)


# # class_counts = train_df["target"].value_counts().sort_index()
# # num_neg = class_counts[0]
# # num_pos = class_counts[1]
# # pos_weight = num_neg / num_pos

# # print("pos_weight:", pos_weight)


# # ======================
# # DataLoaders
# # ======================
 
# cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=32, shuffle=True)
# cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=32, shuffle=False)

# resnet_train_loader = DataLoader(resnet_train_dataset, batch_size=32, shuffle=True)
# resnet_val_loader = DataLoader(resnet_val_dataset, batch_size=32, shuffle=False)




# # ======================
# # Validation
# # ======================
# def validate(model, val_loader, device, criterion):
#     model.eval()
#     val_loss = 0.0

#     all_labels = []
#     all_preds = []
#     all_probs = []

#     print("🔍 Validation started...")

#     with torch.no_grad():
#         for batch_idx, (images, labels) in enumerate(val_loader):
#             images = images.to(device, non_blocking=True)
#             labels = labels.to(device).float().unsqueeze(1)

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             val_loss += loss.item()

#             probs = torch.sigmoid(outputs)
#             preds = (probs >= 0.5).float()

#             all_labels.extend(labels.cpu().numpy().flatten())
#             all_probs.extend(probs.cpu().numpy().flatten()) 
#             all_preds.extend(preds.cpu().numpy().flatten())

#             if batch_idx % 100 == 0:
#                 print(f"Val Batch {batch_idx}/{len(val_loader)} | Loss: {loss.item():.4f}")

#     val_loss = val_loss / len(val_loader)

#     val_acc = accuracy_score(all_labels, all_preds)
#     val_f1 = f1_score(all_labels, all_preds, zero_division=0)
#     val_recall = recall_score(all_labels, all_preds, zero_division=0)
#     val_precision = precision_score(all_labels, all_preds, zero_division=0)

#     print("✅ Validation done")

#     return val_loss, val_acc, val_f1, val_recall, val_precision
# # ======================
# # Training
# # ======================
# def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001, save_path="model_best.pth"):
#     model = model.to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     # criterion = nn.BCEWithLogitsLoss(
#     # pos_weight=torch.tensor([math.sqrt(pos_weight)], dtype=torch.float32, device=device)
#     # )
#     criterion = nn.BCEWithLogitsLoss()

#     train_losses = []
#     val_losses = []
#     val_accuracies = []
#     val_f1s = []
#     val_recalls = []
#     val_precisions = []

#     best_val_f1 = 0.0

#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0

#         print(f"\n🚀 Starting Epoch {epoch+1}/{epochs}")

#         for batch_idx, (images, labels) in enumerate(train_loader):
#             images = images.to(device)
#             labels = labels.to(device).float().unsqueeze(1)

#             optimizer.zero_grad()

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#             if batch_idx % 100 == 0:
#                 print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

#         epoch_train_loss = running_loss / len(train_loader)

#         print("➡️ Running validation...")
#         epoch_val_loss, val_acc, val_f1, val_recall, val_precision = validate(
#             model, val_loader, device, criterion
#         )

#         train_losses.append(epoch_train_loss)
#         val_losses.append(epoch_val_loss)
#         val_accuracies.append(val_acc)
#         val_f1s.append(val_f1)
#         val_recalls.append(val_recall)
#         val_precisions.append(val_precision)

#         if val_f1 > best_val_f1:
#             best_val_f1 = val_f1
#             torch.save(model.state_dict(), save_path)
#             print(f"💾 Best model saved to {save_path} | Val F1: {val_f1:.4f}")

#         print(
#             f"📊 Epoch {epoch+1}/{epochs} DONE | "
#             f"Train Loss: {epoch_train_loss:.4f} | "
#             f"Val Loss: {epoch_val_loss:.4f} | "
#             f"Val Acc: {val_acc:.4f} | "
#             f"Val F1: {val_f1:.4f} | "
#             f"Val Recall: {val_recall:.4f} | "
#             f"Val Precision: {val_precision:.4f}"
#         )

#     return {
#         "model": model,
#         "train_losses": train_losses,
#         "val_losses": val_losses,
#         "val_accuracies": val_accuracies,
#         "val_f1s": val_f1s,
#         "val_recalls": val_recalls,
#         "val_precisions": val_precisions,
#     }


# # ======================
# # Train CNN
# # ======================
# cnn_model = BaselineCNN()

# cnn_results = train_model(
#     model=cnn_model,
#     train_loader=cnn_train_loader,
#     val_loader=cnn_val_loader,
#     device=device,
#     epochs=10,
#     lr=0.001,
#     save_path="cnn_model_best.pth"
# )

# resnet_model = get_resnet()

# resnet_results = train_model(
#     model=resnet_model,
#     train_loader=resnet_train_loader,
#     val_loader=resnet_val_loader,
#     device=device,
#     epochs=10,
#     lr=0.0001,
#     save_path="resnet_model_best.pth"
# )


# import os
# from pathlib import Path

# import pandas as pd
# import torch
# import torch.nn as nn
# from PIL import Image
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
# from sklearn.model_selection import GroupShuffleSplit
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from tqdm import tqdm

# from dataset import MelanomaDataset
# from baseline_model import get_resnet


# # ======================
# # Device
# # ======================
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

# print("Using device:", device)


# # ======================
# # Resize images once
# # ======================
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

#     for img_name in tqdm(image_files, desc="Resizing images"):
#         input_path = input_dir / img_name
#         output_path = output_dir / img_name

#         if output_path.exists():
#             continue

#         try:
#             img = Image.open(input_path).convert("RGB")
#             img = img.resize(size)
#             img.save(output_path)
#         except Exception as e:
#             print(f"Error processing {img_name}: {e}")

#     print("Resized dataset ready.")


# # ======================
# # Paths
# # ======================
# DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))

# input_dir = DATASET_PATH / "jpeg" / "train"
# output_dir = DATASET_PATH / "jpeg_224" / "train"
# labels_path = DATASET_PATH / "train.csv"

# train_image_dir = output_dir

# if not output_dir.exists() or len(os.listdir(output_dir)) == 0:
#     prepare_resized_images(input_dir, output_dir, size=(224, 224))
# else:
#     print("Resized dataset already exists.")

# # ======================
# # Read CSV
# # ======================
# df = pd.read_csv(labels_path)
# df = df[["image_name", "patient_id", "target"]]


# # ======================
# # Patient-wise split
# # ======================
# gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

# train_df = df.iloc[train_idx].reset_index(drop=True)
# val_df = df.iloc[val_idx].reset_index(drop=True)

# print("Train shape:", train_df.shape)
# print("Val shape:", val_df.shape)

# overlap = set(train_df["patient_id"]).intersection(set(val_df["patient_id"]))
# print("Overlapping patients:", len(overlap))


# # ======================
# # Transforms
# # ======================
# train_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# val_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])


# # ======================
# # Datasets and loaders
# # ======================
# train_dataset = MelanomaDataset(train_df, train_image_dir, transform=train_transform)
# val_dataset = MelanomaDataset(val_df, train_image_dir, transform=val_transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# # ======================
# # Validation
# # ======================
# def validate(model, val_loader, device, criterion):
#     model.eval()
#     total_loss = 0.0

#     all_labels = []
#     all_probs = []
#     all_preds = []

#     with torch.no_grad():
#         for images, labels in val_loader:
#             images = images.to(device)
#             labels = labels.to(device).unsqueeze(1)

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             probs = torch.sigmoid(outputs)
#             preds = (probs >= 0.5).float()

#             total_loss += loss.item() * images.size(0)

#             all_labels.extend(labels.cpu().numpy().flatten())
#             all_probs.extend(probs.cpu().numpy().flatten())
#             all_preds.extend(preds.cpu().numpy().flatten())

#     val_loss = total_loss / len(val_loader.dataset)
#     val_acc = accuracy_score(all_labels, all_preds)
#     val_f1 = f1_score(all_labels, all_preds, zero_division=0)
#     val_recall = recall_score(all_labels, all_preds, zero_division=0)
#     val_precision = precision_score(all_labels, all_preds, zero_division=0)

#     try:
#         val_auc = roc_auc_score(all_labels, all_probs)
#     except ValueError:
#         val_auc = 0.0

#     return val_loss, val_acc, val_f1, val_recall, val_precision, val_auc


# # ======================
# # Training
# # ======================
# def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4, save_path="resnet18_baseline_best.pth"):
#     model = model.to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.BCEWithLogitsLoss()

#     best_val_auc = 0.0

#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0

#         print(f"\nStarting epoch {epoch + 1}/{epochs}")

#         for batch_idx, (images, labels) in enumerate(train_loader):
#             images = images.to(device)
#             labels = labels.to(device).unsqueeze(1)

#             optimizer.zero_grad()

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * images.size(0)

#             if batch_idx % 100 == 0:
#                 print(f"Epoch {epoch + 1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

#         train_loss = running_loss / len(train_loader.dataset)

#         val_loss, val_acc, val_f1, val_recall, val_precision, val_auc = validate(
#             model, val_loader, device, criterion
#         )

#         print(f"Epoch {epoch + 1}/{epochs}")
#         print(f"Train Loss:    {train_loss:.4f}")
#         print(f"Val Loss:      {val_loss:.4f}")
#         print(f"Val Accuracy:  {val_acc:.4f}")
#         print(f"Val F1:        {val_f1:.4f}")
#         print(f"Val Recall:    {val_recall:.4f}")
#         print(f"Val Precision: {val_precision:.4f}")
#         print(f"Val ROC-AUC:   {val_auc:.4f}")

#         if val_auc > best_val_auc:
#             best_val_auc = val_auc
#             torch.save(model.state_dict(), save_path)
#             print(f"Best model saved to {save_path}")

#     print(f"\nTraining finished. Best Val ROC-AUC: {best_val_auc:.4f}")


# # ======================
# # Run training
# # ======================
# model = get_resnet()

# train_model(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     device=device,
#     epochs=10,
#     lr=1e-4,
#     save_path="resnet18_baseline_best.pth"
# )









# import os
# import pandas as pd
# import torch
# import torch.nn as nn
# from pathlib import Path
# from PIL import Image
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from sklearn.model_selection import GroupShuffleSplit
# from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# from dataset import MelanomaDataset
# from baseline_model import BaselineCNN


# # ======================
# # Device
# # ======================
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

# print("Using device:", device)


# # ======================
# # Resize images once
# # ======================
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
#         input_path = input_dir / img_name
#         output_path = output_dir / img_name

#         if output_path.exists():
#             continue

#         try:
#             img = Image.open(input_path).convert("RGB")
#             img = img.resize(size)
#             img.save(output_path)
#         except Exception as e:
#             print(f"Error with {img_name}: {e}")

#     print("Resized dataset ready.")


# # ======================
# # Paths
# # ======================
# DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))

# input_dir = DATASET_PATH / "jpeg" / "train"
# output_dir = DATASET_PATH / "jpeg_224" / "train"
# train_path = DATASET_PATH / "jpeg_224" / "train"
# labels_path = DATASET_PATH / "train.csv"

# if not output_dir.exists() or len(os.listdir(output_dir)) == 0:
#     prepare_resized_images(input_dir, output_dir, size=(224, 224))
# else:
#     print("Resized dataset already exists.")


# # ======================
# # Read dataframe
# # ======================
# df = pd.read_csv(labels_path)
# df = df[["image_name", "patient_id", "target"]]


# # ======================
# # Patient-wise split
# # ======================
# gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

# train_df = df.iloc[train_idx].reset_index(drop=True)
# val_df = df.iloc[val_idx].reset_index(drop=True)

# print("Train shape:", train_df.shape)
# print("Val shape:", val_df.shape)

# overlap = set(train_df["patient_id"]).intersection(set(val_df["patient_id"]))
# print("Overlapping patients:", len(overlap))


# # ======================
# # Imbalance
# # ======================
# class_counts = train_df["target"].value_counts().sort_index()
# num_neg = class_counts[0]
# num_pos = class_counts[1]
# pos_weight = num_neg / num_pos

# print("Imbalance ratio:", pos_weight)


# # ======================
# # Transforms
# # ======================
# train_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])

# val_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])


# # ======================
# # Datasets
# # ======================
# cnn_train_dataset = MelanomaDataset(train_df, train_path, train_transform)
# cnn_val_dataset = MelanomaDataset(val_df, train_path, val_transform)


# # ======================
# # DataLoaders
# # ======================
# cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=32, shuffle=True)
# cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=32, shuffle=False)


# # ======================
# # Validation
# # ======================
# def validate(model, val_loader, device, criterion):
#     model.eval()
#     val_loss = 0.0

#     all_labels = []
#     all_preds = []

#     print("Validation started...")

#     with torch.no_grad():
#         for batch_idx, (images, labels) in enumerate(val_loader):
#             images = images.to(device, non_blocking=True)
#             labels = labels.to(device).float().unsqueeze(1)

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             val_loss += loss.item()

#             probs = torch.sigmoid(outputs)
#             preds = (probs >= 0.5).float()

#             all_labels.extend(labels.cpu().numpy().flatten())
#             all_preds.extend(preds.cpu().numpy().flatten())

#             if batch_idx % 100 == 0:
#                 print(f"Val Batch {batch_idx}/{len(val_loader)} | Loss: {loss.item():.4f}")

#     val_loss /= len(val_loader)

#     val_acc = accuracy_score(all_labels, all_preds)
#     val_f1 = f1_score(all_labels, all_preds, zero_division=0)
#     val_recall = recall_score(all_labels, all_preds, zero_division=0)
#     val_precision = precision_score(all_labels, all_preds, zero_division=0)

#     print("Validation done")

#     return val_loss, val_acc, val_f1, val_recall, val_precision


# # ======================
# # Training
# # ======================
# def train_model(model, train_loader, val_loader, device, pos_weight, epochs=10, lr=0.001, save_path="cnn_model_best.pth"):
#     model = model.to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.BCEWithLogitsLoss(
#         pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
#     )

#     best_val_f1 = 0.0

#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0

#         print(f"\nStarting Epoch {epoch+1}/{epochs}")

#         for batch_idx, (images, labels) in enumerate(train_loader):
#             images = images.to(device, non_blocking=True)
#             labels = labels.to(device).float().unsqueeze(1)

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#             if batch_idx % 100 == 0:
#                 print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

#         epoch_train_loss = running_loss / len(train_loader)

#         print("Running validation...")
#         epoch_val_loss, val_acc, val_f1, val_recall, val_precision = validate(
#             model, val_loader, device, criterion
#         )

#         if val_f1 > best_val_f1:
#             best_val_f1 = val_f1
#             torch.save(model.state_dict(), save_path)
#             print(f"Best model saved to {save_path} | Val F1: {val_f1:.4f}")

#         print(
#             f"Epoch {epoch+1}/{epochs} DONE | "
#             f"Train Loss: {epoch_train_loss:.4f} | "
#             f"Val Loss: {epoch_val_loss:.4f} | "
#             f"Val Acc: {val_acc:.4f} | "
#             f"Val F1: {val_f1:.4f} | "
#             f"Val Recall: {val_recall:.4f} | "
#             f"Val Precision: {val_precision:.4f}"
#         )


# # ======================
# # Train CNN baseline
# # ======================
# cnn_model = BaselineCNN()

# train_model(
#     model=cnn_model,
#     train_loader=cnn_train_loader,
#     val_loader=cnn_val_loader,
#     device=device,
#     pos_weight=pos_weight,
#     epochs=10,
#     lr=0.001,
#     save_path="cnn_model_best.pth"
# )


import os
import math
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from dataset import MelanomaDataset
from baseline_model import BaselineCNN, get_resnet18_model


# ======================
# Device
# ======================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ======================
# Resize images once
# ======================
def prepare_resized_images(input_dir, output_dir, size=(224, 224)):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    print(f"Total images found: {len(image_files)}")

    for img_name in tqdm(image_files):
        input_path = input_dir / img_name
        output_path = output_dir / img_name

        if output_path.exists():
            continue

        try:
            img = Image.open(input_path).convert("RGB")
            img = img.resize(size)
            img.save(output_path)
        except Exception as e:
            print(f"Error with {img_name}: {e}")

    print("Resized dataset ready.")



# ======================
# Paths
# ======================
DATASET_PATH = Path(os.getenv("DATASET_PATH", "dataset"))

input_dir = DATASET_PATH / "jpeg" / "train"
output_dir = DATASET_PATH / "jpeg_224" / "train"

train_path = output_dir
labels_path = DATASET_PATH / "train.csv"

if not output_dir.exists() or len(os.listdir(output_dir)) == 0:
    prepare_resized_images(input_dir, output_dir)
else:
    print("Resized dataset already exists.")


# ======================
# Data
# ======================
df = pd.read_csv(labels_path)
df = df[["image_name", "patient_id", "target"]]

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)

print("Train shape:", train_df.shape)
print("Val shape:", val_df.shape)


# ======================
# Imbalance
# ======================
class_counts = train_df["target"].value_counts().sort_index()
num_neg = class_counts[0]
num_pos = class_counts[1]
pos_weight = num_neg / num_pos

print("Imbalance ratio:", pos_weight)




# ======================
# Transforms
# ======================

# CNN → simple
cnn_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ResNet → augmentation
resnet_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ======================
# Datasets
# ======================
cnn_train_dataset = MelanomaDataset(train_df, train_path, cnn_transform)
cnn_val_dataset = MelanomaDataset(val_df, train_path, val_transform)

resnet_train_dataset = MelanomaDataset(train_df, train_path, resnet_transform)
resnet_val_dataset = MelanomaDataset(val_df, train_path, val_transform)


# ======================
# Loaders
# ======================
cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=32, shuffle=True)
cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=32)

resnet_train_loader = DataLoader(resnet_train_dataset, batch_size=32, shuffle=True)
resnet_val_loader = DataLoader(resnet_val_dataset, batch_size=32)



def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0

    all_labels = []
    all_preds = []
    all_probs = []

    print("🔍 Validation started...")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.2).float()

            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())

            if batch_idx % 100 == 0:
                print(f"Val Batch {batch_idx}/{len(val_loader)} | Loss: {loss.item():.4f}")

    val_loss = val_loss / len(val_loader)

    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)
    val_recall = recall_score(all_labels, all_preds, zero_division=0)
    val_precision = precision_score(all_labels, all_preds, zero_division=0)

    print("✅ Validation done")

    return val_loss, val_acc, val_f1, val_recall, val_precision





def train_model(model, train_loader, val_loader, device, pos_weight, epochs=10, lr=0.001, save_path="model_best.pth"):
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )

    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1s = []
    val_recalls = []
    val_precisions = []

    best_val_f1 = -1.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        print(f"\n🚀 Starting Epoch {epoch+1}/{epochs}")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        epoch_train_loss = running_loss / len(train_loader)

        print("➡️ Running validation...")
        epoch_val_loss, val_acc, val_f1, val_recall, val_precision = validate(
            model, val_loader, device, criterion
        )
        

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)
        val_recalls.append(val_recall)
        val_precisions.append(val_precision)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"💾 Best model saved to {save_path} | Val F1: {val_f1:.4f}")

        print(
            f"📊 Epoch {epoch+1}/{epochs} DONE | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val Recall: {val_recall:.4f} | "
            f"Val Precision: {val_precision:.4f}"
        )

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "val_f1s": val_f1s,
        "val_recalls": val_recalls,
        "val_precisions": val_precisions,
    }



cnn_results = train_model(
    model=BaselineCNN(),
    train_loader=cnn_train_loader,
    val_loader=cnn_val_loader,
    device=device,
    pos_weight=pos_weight,
    epochs=10,
    lr=0.001,
    save_path="cnn_model_best.pth"
)

import math
resnet_results = train_model(
    model=get_resnet18_model(),
    train_loader=resnet_train_loader,
    val_loader=resnet_val_loader,
    device=device,
    pos_weight=math.sqrt(pos_weight),
    epochs=10,
    lr=0.0001,
    save_path="resnet18_model_best.pth"
)
