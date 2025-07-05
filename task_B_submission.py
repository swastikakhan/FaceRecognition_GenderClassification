import os
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# === Device Setup ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Models ===
mtcnn = MTCNN(keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# === Comment for local users ===
# If you're running this locally (not on Google Colab),
# update the following paths to your local directory
train_dir = "/content/Comsys_Hackathon5/Task_B/train"
val_dir = "/content/Comsys_Hackathon5/Task_B/val"

# === Face Embedding Extraction ===
def get_embedding(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        face = mtcnn(img)
        if face is None:
            return None
        with torch.no_grad():
            emb = facenet(face.unsqueeze(0).to(device))
        return emb.cpu().numpy().flatten()
    except:
        return None

# === Load all embeddings from a directory ===
def load_embeddings(folder):
    embeddings, labels, paths = [], [], []

    persons = os.listdir(folder)
    for person in tqdm(persons, desc=f"Processing {folder}"):
        person_dir = os.path.join(folder, person)
        for root, _, files in os.walk(person_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    emb = get_embedding(img_path)
                    if emb is not None:
                        embeddings.append(emb)
                        labels.append(person)
                        paths.append(img_path)
    return np.array(embeddings), np.array(labels), paths

# === Generate Verification Pairs ===
def generate_pairs(embeddings, labels):
    positives, negatives = [], []
    unique_labels = np.unique(labels)
    label_to_indices = {label: np.where(labels == label)[0] for label in unique_labels}

    for label, indices in label_to_indices.items():
        if len(indices) >= 2:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    positives.append((indices[i], indices[j]))

    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            idx1 = np.random.choice(label_to_indices[unique_labels[i]])
            idx2 = np.random.choice(label_to_indices[unique_labels[j]])
            negatives.append((idx1, idx2))

    return positives, negatives

# === Cosine Similarity ===
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# === Main Pipeline ===
if __name__ == "__main__":
    # Load all embeddings
    X_train, y_train, train_paths = load_embeddings(train_dir)
    X_val, y_val, val_paths = load_embeddings(val_dir)

    # Merge both sets
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    paths_all = train_paths + val_paths

    # Generate positive and negative pairs
    pos_pairs, neg_pairs = generate_pairs(X_all, y_all)
    n_pairs = min(len(pos_pairs), len(neg_pairs))

    pairs = pos_pairs[:n_pairs] + neg_pairs[:n_pairs]
    pair_labels = [1] * n_pairs + [0] * n_pairs

    # Compute similarity scores
    X_sim = np.array([cosine_sim(X_all[i], X_all[j]) for i, j in pairs])

    # Train-test split
    X_train_sim, X_test_sim, y_train_sim, y_test_sim = train_test_split(
        X_sim, pair_labels, test_size=0.2, random_state=42
    )

    # Determine best threshold on train set
    fpr, tpr, thresholds = roc_curve(y_train_sim, X_train_sim)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Predict on test set
    y_pred = (X_test_sim >= optimal_threshold).astype(int)

    # === Final Results ===
    print("\n✅ Final Task B Results (Face Verification)")
    print(f"Top-1 Accuracy: {accuracy_score(y_test_sim, y_pred):.4f}")
    print(f"Macro F1-score: {f1_score(y_test_sim, y_pred, average='macro'):.4f}")
