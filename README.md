FaceRecognition_GenderClassification 🧠
Repository: swastikakhan/FaceRecognition_GenderClassification

📌 Overview
This codebase contains models for both Task A (Gender Classification) and Task B (Face Recognition) under challenging visual conditions, using the FACECOM dataset:

Task A: Binary gender classification (Male/Female).

Task B: Multi-class face recognition across identities.

Data perturbations: Blur, fog, low-light, rain, overexposure.

📁 Repository Structure
bash
Copy
Edit
/
├── TaskA_98.ipynb            # Gender Classification Jupyter notebook (98% accuracy)
├── TaskB_88.ipynb            # Face Recognition notebook (~88% accuracy)
├── male_female_augmented.ipynb  # Data augmentation script for Task A
├── requirements.txt          # Python dependencies
└── README.md                 # (this file)
📥 Dataset
FACECOM dataset download:
Link: https://drive.google.com/file/d/1LgjPFk7tgCRhJVfL8SCKX9Z1N2wK4_6_/view

Directory structure:

bash
Copy
Edit
data/
├── TaskA/
│   ├── train/male/…, train/female/
│   └── val/male/…, val/female/
└── TaskB/
    ├── <person_id_1>/…      # face images per identity
    ├── <person_id_2>/…
    └── distorted/…          # perturbed test images
🧠 Methodology
Task A – Gender Classification
Notebook: TaskA_98.ipynb

Process:

Load and split dataset.

Apply augmentations (e.g., blur, brightness, contrast).

Train using a CNN backbone (e.g., ResNet or EfficientNet).

Metrics: Accuracy, Precision, Recall, F1‑Score.

Achieved ~98% accuracy on validation.

Task B – Face Recognition
Notebook: TaskB_88.ipynb

Process:

Load multi-class dataset with identity labels.

Use CNN backbone + ArcFace/CosFace (softmax with margin).

Train and evaluate: Top-1 Accuracy, macro F1‑score.

Achieved ~88% Top-1 accuracy under distorted conditions.

Data Augmentation
Script: male_female_augmented.ipynb

Techniques include:

Motion blur, fog simulation

Additive noise, uneven lighting

Hue/contrast variations

🚀 Getting Started
Setup
bash
Copy
Edit
git clone https://github.com/swastikakhan/FaceRecognition_GenderClassification.git
cd FaceRecognition_GenderClassification
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
How to Run
Download and unzip the FACECOM dataset into the data/ directory.

Launch notebooks:

bash
Copy
Edit
jupyter notebook TaskA_98.ipynb
jupyter notebook TaskB_88.ipynb
Inspect male_female_augmented.ipynb for augmentation strategies.

📊 Results
Task	Validation Accuracy
Gender Classification (Task A)	98 %
Face Recognition (Task B)	88 % Top‑1

