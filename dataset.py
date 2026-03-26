

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
from PIL import Image
from torch.utils.data import Dataset


class MelanomaDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_name = row["image_name"]
        label = int(row["target"])

        image_path = self.image_dir / f"{image_name}.jpg"
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label