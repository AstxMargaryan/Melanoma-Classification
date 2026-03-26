import os
from PIL import Image
from torch.utils.data import Dataset


class MelanomaDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        """
        Args:
            df (pd.DataFrame): dataframe with image_name and target
            image_dir (str or Path): path to image folder
            transform (callable, optional): image transformations
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get row
        row = self.df.iloc[idx]

        # Image path
        img_path = os.path.join(self.image_dir, row["image_name"] + ".jpg")
        if idx == 0:
            print("DEBUG PATH:", img_path)

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Target
        label = row["target"]

        return image, label