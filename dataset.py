import os
from PIL import Image
from torch.utils.data import Dataset
import torch


class MelanomaDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.loc[idx, "image_name"]
        label = self.df.loc[idx, "target"]

        image_path = os.path.join(self.image_dir, image_name + ".jpg")
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)



