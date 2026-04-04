# 🔬 SIIM-ISIC Melanoma Classification

<div align="center">

**Deep Learning pipeline for automated melanoma detection from dermoscopic skin lesion images.**
</div>

## 🧬 What is Melanoma?

Melanoma is the deadliest form of skin cancer, responsible for 75% of all skin cancer deaths despite being the least common type. It occurs when pigment-making cells in the skin, called melanocytes, begin to reproduce uncontrollably. Melanoma can form from an existing mole or develop on unblemished skin.
<div align="center">
<img src="media/image1.png" width="500">
</div>

The most common type of melanoma spreads on the skin's surface. It is called superficial spreading melanoma. It may stay on the surface or grow down into deeper tissues. Other types of melanoma can start anywhere on or inside the body, including under fingernails or toenails and inside the eye.


## 📋 Project Overview

This project tackles the [SIIM-ISIC Melanoma Classification Kaggle Competition](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/overview): given a dermoscopic image of a skin lesion, predict the probability that it is **malignant (melanoma)** or **benign**.


### Dataset

| Property | Details |
|---|---|
| Organizer | SIIM + International Skin Imaging Collaboration (ISIC) |
| Training images | 33,126  |
| Test images | 10,982 |
| Format | JPEG / DICOM |
| Labels | Binary — `0` benign · `1` malignant (melanoma) |
| Patient metadata | Age, sex, anatomical site |
| **Class split** | **~98.2% benign / ~1.8% malignant** |
| Evaluation metric | **ROC-AUC** |


### CSV Columns

| Column | Description | Split |
|---|---|---|
| `image_name` | Unique identifier — filename of the related DICOM image | train + test |
| `patient_id` | Unique patient identifier | train + test |
| `sex` | Sex of the patient *(may be missing)* | train + test |
| `age_approx` | Approximate patient age at time of imaging | train + test |
| `anatom_site_general_challenge` | Anatomical location of the lesion | train + test |
| `diagnosis` | Detailed diagnosis label | train only |
| `benign_malignant` | Indicator of malignancy | train only |
| `target` | Binary target — `0` benign · `1` malignant | train only |


> ⚠️ **Class Imbalance:** A naive model that always predicts "benign" achieves 98% accuracy — but AUC of only 0.5. This is why we use AUC as the metric and `BCEWithLogitsLoss` with a computed `pos_weight` (~49×) to heavily penalize missed melanomas.
The dataset consists of dermoscopic images of skin lesions along with associated metadata. Images are provided in DICOM format (as well as JPEG and TFRecord formats), while additional information is available in CSV files.





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