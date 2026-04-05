import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import cv2

class MelanomaDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.loc[idx, "image_name"]
        target = self.df.loc[idx, "target"]

        img_path = os.path.join(self.img_dir, image_name + ".jpg")

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        label = torch.tensor(target, dtype=torch.float32)


        return image, label

