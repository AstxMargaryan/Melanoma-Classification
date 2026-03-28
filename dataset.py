

#     def __getitem__(self, idx):
#         # Get row
#         row = self.df.iloc[idx]

#         # Image path
#         img_path = os.path.join(self.image_dir, row["image_name"] + ".jpg")
#         if idx == 0:
#             print("DEBUG PATH:", img_path)

#         # Load image
#         image = Image.open(img_path).convert("RGB")

#         # Apply transforms
#         if self.transform:
#             image = self.transform(image)

#         # Target
#         label = row["target"]

#         return image, label


# import os
# from PIL import Image
# from torch.utils.data import Dataset


# class MelanomaDataset(Dataset):
#     def __init__(self, df, image_dir, transform=None):
#         self.df = df.reset_index(drop=True)
#         self.image_dir = image_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         image_name = self.df.loc[idx, "image_name"]
#         label = self.df.loc[idx, "target"]

#         image_path = os.path.join(self.image_dir, image_name + ".jpg")

#         image = Image.open(image_path).convert("RGB")

#         if self.transform:
#             image = self.transform(image)

#         return image, label


# import os
# from PIL import Image
# from torch.utils.data import Dataset


# class MelanomaDataset(Dataset):
#     def __init__(self, df, image_dir, transform=None):
#         self.df = df.reset_index(drop=True)
#         self.image_dir = image_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         image_name = self.df.loc[idx, "image_name"]
#         label = int(self.df.loc[idx, "target"])

#         image_path = os.path.join(self.image_dir, image_name + ".jpg")

#         image = Image.open(image_path).convert("RGB")

#         if self.transform:
#             image = self.transform(image)

#         return image, label


from pathlib import Path

from matplotlib import transforms
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

# class MelanomaDataset(Dataset):
#     def __init__(self, df, img_dir, transform=None):
#         self.df = df
#         self.img_dir = img_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         img_name = self.df.iloc[idx]['image_name'] + ".jpg"
#         img_path = os.path.join(self.img_dir, img_name)

#         image = Image.open(img_path).convert("RGB")
#         label = self.df.iloc[idx]['target']

#         if self.transform:
#             image = self.transform(image)

#         return image, label




import os
import cv2
import torch
from torch.utils.data import Dataset

class MelanomaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = str(img_dir)
        self.transform = transform

        # արագության համար
        self.image_names = self.df["image_name"].values
        self.labels = self.df["target"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        label = self.labels[idx]

        img_path = os.path.join(self.img_dir, img_name + ".jpg")

        # read image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # transforms (albumentations style)
        if self.transform:
            image = self.transform(image=image)["image"]

        # 🔥 ՍԱ Է ՃԻՇՏԸ
        label = torch.tensor(label, dtype=torch.long)

        return image, label

