import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import get_model


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def load_config(config_path="models/best_model_config.json"):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Make sure you already ran training and saved best_model_config.json"
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    required_keys = ["model_name", "checkpoint", "threshold"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing key '{key}' in {config_path}")

    return config


def build_transform(image_size=224):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


def load_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    return image


def load_model(model_name, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = get_model(model_name)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def predict(image_path, config_path="models/best_model_config.json"):
    config = load_config(config_path)

    model_name = config["model_name"]
    checkpoint_path = config["checkpoint"]
    threshold = float(config["threshold"])

    image = load_image(image_path)
    transform = build_transform(image_size=224)
    transformed = transform(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    model = load_model(model_name, checkpoint_path, DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()

    label = "Malignant" if probability >= threshold else "Benign"
    confidence = probability if label == "Malignant" else (1 - probability)

    print(f"Prediction: {label} | Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        predict(image_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)