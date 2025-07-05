# COMSYS Hackathon5 – Final Submission

This repository contains the complete solution for both Task A and Task B of COMSYS Hackathon 5.

---

## How to Run

### Task A – Multi-class Face Recognition

- **File**: `task_A_submission.py`
- **To run**:
  1. Place the dataset in the required folder structure.
  2. Change the `train_dir` and `val_dir` variables at the top of the script if necessary.
  3. Execute the script:

     ```bash
     python task_A_submission.py
     ```

- **Outputs**:
  - Final evaluation metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
  - Best model saved as: `best_gender_model.pt`

---

### Task B – Face Verification

- **File**: `task_B_submission.py`
- **To run**:
  1. Make sure the dataset is structured as `train/` and `val/` directories.
  2. Set the correct paths in the script (`train_dir`, `val_dir`).
  3. Run the script:

     ```bash
     python task_B_submission.py
     ```

- **Outputs**:
  - Final evaluation metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score

> This script does not train a model. It uses a pretrained FaceNet-based embedding extractor and performs verification using cosine similarity with an ROC-optimized threshold.

---

## Model Architectures

### Task A

- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Classifier Head**: Replaced `_fc` with `nn.Linear(in_features=1280, out_features=2)`
- **Loss Function**: CrossEntropyLoss; Optimizer: Adam (lr=2e-4); Scheduler: CosineAnnealingLR
- **Best model saved** as `best_gender_model.pt` based on validation accuracy


### Task B

- **Face Detection**: MTCNN
- **Embedding Model**: InceptionResnetV1 (pretrained)
- **Verification Method**:
  - Cosine similarity is computed between face embeddings.
  - A similarity threshold is selected using the ROC curve from training pairs.
  - Final predictions are made based on whether a pair’s similarity exceeds the threshold.

> No model weights are saved in Task B, as no classifier is trained.

- At the time of model development, it was not specified that separate training and testing metrics would be required. Due to time constraints during submission, we are reporting the evaluation results obtained from our combined training and validation process.
- However, in the submitted code, we have updated the evaluation metrics for ease of verification.
---

## Requirements

Install all dependencies using:

```bash
pip install torch torchvision facenet-pytorch efficientnet-pytorch scikit-learn numpy pillow tqdm matplotlib

