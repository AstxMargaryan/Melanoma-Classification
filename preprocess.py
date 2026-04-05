import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

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
            img = Image.open(input_path)
            img = img.convert("RGB")
            img = img.resize(size)
            img.save(output_path)
        except Exception as e:
            print(f"Error with {img_name}: {e}")

    print("Resized dataset ready.")