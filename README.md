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


## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/AstxMargaryan/Melanoma-Classification.git
cd Melanoma-Classification
```


### 2. Create virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```
###  3. Install dependencies

```bash
pip install -r requirements.txt
```

##  4. Dataset Setup

The dataset is NOT included in this repository.

### Option 1 — Download manually (Recommended)

Download the dataset from Google Drive:

👉 [Download Dataset](https://drive.google.com/drive/folders/1y-LUVhRqnVz1XseVpAkFJ3PYm9Uu6-WB?usp=drive_link)

Then extract it into the project root so the structure looks like:

```bash
Melanoma-Classification/
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── train/
│   ├── test/
│   ├── jpeg/
│   └── tfrecords/
```

### Option 2 — Use Google Colab

If running in Google Colab:

```bash
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ["DATASET_PATH"] = "/content/drive/MyDrive/dataset"
```