import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import get_model


# =====================================================
# DEVICE
# =====================================================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# =====================================================
# INFERENCE PREPROCESSING
# Raw input image -> resize to 224x224 -> normalize -> tensor
# =====================================================
def get_inference_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


# =====================================================
# LOAD IMAGE
# =====================================================
def load_image(image_path: str) -> np.ndarray:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    return np.array(image)


# =====================================================
# LOAD MODEL
# =====================================================
def load_model(model_name: str, checkpoint_path: str):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = get_model(model_name)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    return model


# =====================================================
# SINGLE PREDICTION
# =====================================================
def predict_image(model, image_path: str, threshold: float):
    image = load_image(image_path)
    transform = get_inference_transform()

    transformed = transform(image=image)
    tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        malignant_prob = torch.sigmoid(logits).item()

    pred_class = 1 if malignant_prob >= threshold else 0
    pred_label = "Malignant (Melanoma)" if pred_class == 1 else "Benign"

    confidence = malignant_prob if pred_class == 1 else (1.0 - malignant_prob)

    if malignant_prob >= 0.75:
        risk_level = "High Risk"
    elif malignant_prob >= 0.40:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    return {
        "prediction": pred_label,
        "confidence": confidence,
        "malignant_probability": malignant_prob,
        "threshold": threshold,
        "risk_level": risk_level
    }


# =====================================================
# OPTIONAL TTA PREDICTION
# =====================================================
def predict_image_tta(model, image_path: str, threshold: float):
    image = load_image(image_path)

    transforms = [
        lambda x: x,
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: np.rot90(x, 1).copy(),
        lambda x: np.rot90(x, 3).copy(),
    ]

    probs = []
    base_transform = get_inference_transform()

    with torch.no_grad():
        for aug_fn in transforms:
            aug_image = aug_fn(image)
            transformed = base_transform(image=aug_image)
            tensor = transformed["image"].unsqueeze(0).to(DEVICE)

            logits = model(tensor)
            prob = torch.sigmoid(logits).item()
            probs.append(prob)

    malignant_prob = float(np.mean(probs))

    pred_class = 1 if malignant_prob >= threshold else 0
    pred_label = "Malignant (Melanoma)" if pred_class == 1 else "Benign"
    confidence = malignant_prob if pred_class == 1 else (1.0 - malignant_prob)

    if malignant_prob >= 0.75:
        risk_level = "High Risk"
    elif malignant_prob >= 0.40:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    return {
        "prediction": pred_label,
        "confidence": confidence,
        "malignant_probability": malignant_prob,
        "threshold": threshold,
        "risk_level": risk_level
    }


# =====================================================
# MAIN
# =====================================================
def main():
    parser = argparse.ArgumentParser(
        description="Standalone inference script for melanoma classification"
    )

    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to a single input image"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["cnn", "resnet50", "efficientnet_b3"],
        help="Model architecture used during training"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to saved model weights (.pth)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Best threshold selected after validation"
    )

    parser.add_argument(
        "--tta",
        action="store_true",
        help="Use Test Time Augmentation"
    )

    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Loading model: {args.model_name}")
    print(f"Loading checkpoint: {args.checkpoint}")

    model = load_model(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint
    )

    if args.tta:
        result = predict_image_tta(
            model=model,
            image_path=args.image_path,
            threshold=args.threshold
        )
    else:
        result = predict_image(
            model=model,
            image_path=args.image_path,
            threshold=args.threshold
        )

    print("\n" + "=" * 50)
    print("🧠 MELANOMA CLASSIFICATION RESULT")
    print("=" * 50)
    print(f"📷 Image: {args.image_path}")
    print(f"🧬 Model: {args.model_name}")
    print(f"✨ TTA: {'Enabled' if args.tta else 'Disabled'}")

    print("\n🔍 Prediction:")
    print(f"   ➤ Class: {result['prediction']}")

    print("\n📊 Confidence:")
    print(f"   ➤ Confidence: {result['confidence'] * 100:.2f}%")

    print("\n📈 Details:")
    print(f"   ➤ Malignant probability: {result['malignant_probability'] * 100:.2f}%")
    print(f"   ➤ Threshold used: {result['threshold']:.2f}")

    print("\n⚠️ Risk Assessment:")
    print(f"   ➤ {result['risk_level']}")

    print("=" * 50)


if __name__ == "__main__":
    main()