# Melanoma-Classification
This project develops a deep learning model to detect melanoma from dermoscopic images by incorporating patient-level contextual information, aiming to improve early diagnosis and support clinical decision-making.




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

👉 [Download Dataset](https://drive.google.com/drive/folders/1RVukTxOQh0fXm3kpTRPTFHZ8ELBN-nsW?usp=drive_link)

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