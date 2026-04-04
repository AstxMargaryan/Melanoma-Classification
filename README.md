# Melanoma-Classification

## Project Overview

Skin cancer is the most common type of cancer worldwide, and one of its most dangerous forms is melanoma. Although melanoma is relatively rare, it is responsible for approximately 75% of skin cancer-related deaths. Melanoma is a life-threatening disease, but when detected early, most cases can be successfully treated with minor surgery.

Detecting melanoma is a challenging task, as the differences between skin lesions are often subtle and difficult to identify even for experienced dermatologists. Artificial intelligence enables the analysis of medical images and the detection of hidden patterns, improving diagnostic accuracy and identifying features that may not be noticeable to dermatologists.

The goal of this project is to develop a deep learning model that analyzes dermoscopic images of skin lesions and predicts the probability (0–1) of whether they are malignant (melanoma) or benign, with higher values indicating greater melanoma risk.

The dataset consists of dermoscopic images of skin lesions along with associated metadata. Images are provided in DICOM format (as well as JPEG and TFRecord formats), while additional information is available in CSV files.

Each sample includes the following features:

- `image_name` – unique identifier, points to filename of related DICOM image  
- `patient_id` – unique patient identifier  
- `sex` – sex of the patient (may be missing)  
- `age_approx` – approximate patient age at time of imaging  
- `anatom_site_general_challenge` – anatomical location of the lesion  
- `diagnosis` – detailed diagnosis (train only)  
- `benign_malignant` – indicator of malignancy  
- `target` – binary target (0 = benign, 1 = malignant)  

This problem is formulated as a probabilistic binary classification task, where the model estimates the probability that a given skin lesion is malignant.


## ⚙️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/AstxMargaryan/Melanoma-Classification.git
cd Melanoma-Classification
```


### 2. Create virtual environment

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

###  3. Install dependencies

```bash
pip install -r requirements.txt
```

## 📦 Dataset Setup

The dataset is NOT included in this repository.

### 🖥️ Option 1 — Local Setup

Download the dataset:

📁 [Download Dataset](https://drive.google.com/drive/folders/1y-LUVhRqnVz1XseVpAkFJ3PYm9Uu6-WB)

Extract it into the project root:

```
Melanoma-Classification/
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── train/
│   ├── test/
│   ├── jpeg/
│   │   ├── train/           ← original SIIM-ISIC training images
│   │   └── test/            ← original SIIM-ISIC test images
│   │
│   │   ── External ──────────────────────────────
│   ├── train-metadata.csv   ← external dataset labels
│   └── image/               ← external ISIC images
```

Run training:

```bash
python3 train.py
```

### ☁️ Option 2 — Google Colab (Recommended — free T4 GPU🚀)


**Step 1 — Mount Google Drive**
```bash
from google.colab import drive
drive.mount('/content/drive')
```
**Step 2 — Prepare dataset**

```bash
# Extract pre-resized images
!unzip "/content/drive/MyDrive/dataset/jpeg_224.zip" -d /content/ -x "__MACOSX/*"

# Extract external images
!unzip "/content/drive/MyDrive/dataset/image.zip" -d /content/ -x "__MACOSX/*"

# Copy CSV files
!cp /content/drive/MyDrive/dataset/train.csv /content/train.csv
!cp /content/drive/MyDrive/dataset/train-metadata.csv /content/train-metadata.csv
```

**Step 3 — Clone the repository**

```bash
!git clone https://github.com/AstxMargaryan/Melanoma-Classification.git
%cd /content/Melanoma-Classification
```
**Step 4 — Install dependencies**

```bash
!pip install -r requirements.txt
```

**Step 5 — Set dataset path**

```python
import os
os.environ["DATASET_PATH"] = "/content"
```
**Step 5 — Run training**
```bash
!python3 train.py
```