import os
import zipfile
import numpy as np
import torch
import joblib
from PIL import Image, ImageEnhance
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

# --- Device setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Models ---
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
              thresholds=[0.4, 0.5, 0.5], device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- Embedding extraction ---
def get_embedding(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        if img.size[0] < 160 or img.size[1] < 160:
            img = img.resize((160, 160), Image.Resampling.LANCZOS)
        face = mtcnn(img)
        if face is None:
            enhancer = ImageEnhance.Contrast(img)
            face = mtcnn(enhancer.enhance(2.0))
        if face is None:
            return None
        with torch.no_grad():
            emb = facenet(face.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            return emb / np.linalg.norm(emb)
    except:
        return None

# --- Dataset loading ---
def load_dataset(folder):
    X, y = [], []
    for person in sorted(os.listdir(folder)):
        person_dir = os.path.join(folder, person)
        if not os.path.isdir(person_dir):
            continue
        for root, _, files in os.walk(person_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)
                    emb = get_embedding(img_path)
                    if emb is not None:
                        X.append(emb)
                        y.append(person)
    return np.array(X), np.array(y)

# --- Path setup (edit this as needed) ---
train_dir = '/content/Comsys_Hackathon5/Task_A/train'
val_dir = '/content/Comsys_Hackathon5/Task_A/val'

# --- Load data ---
X_train, y_train = load_dataset(train_dir)
X_val, y_val = load_dataset(val_dir)

# --- Label encoding ---
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)

# --- Model training ---
models = {}

# 1. Linear SVM
svm_linear = SVC(kernel='linear', probability=True, class_weight='balanced')
svm_linear.fit(X_train, y_train_enc)
models['Linear SVM'] = svm_linear

# 2. RBF SVM
param_grid = {
    'C': [1, 10],
    'gamma': ['scale', 0.01],
    'class_weight': ['balanced']
}
svm_rbf = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid,
                       cv=3, scoring='accuracy', n_jobs=-1)
svm_rbf.fit(X_train, y_train_enc)
models['RBF SVM'] = svm_rbf.best_estimator_

# 3. Random Forest
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_enc)
models['Random Forest'] = rf

# 4. MLP
mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=500, random_state=42)
mlp.fit(X_train, y_train_enc)
models['MLP'] = mlp

# --- Select best model based on validation accuracy ---
best_model = None
best_acc = 0
for name, model in models.items():
    preds = model.predict(X_val)
    acc = accuracy_score(y_val_enc, preds)
    if acc > best_acc:
        best_model = model
        best_preds = preds
        best_acc = acc

# --- Final metrics ---
accuracy = accuracy_score(y_val_enc, best_preds)
precision = precision_score(y_val_enc, best_preds, average='weighted')
recall = recall_score(y_val_enc, best_preds, average='weighted')
f1 = f1_score(y_val_enc, best_preds, average='weighted')

print("Final Evaluation (Task A)")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# --- Save model and label encoder ---
joblib.dump(best_model, 'best_face_recognition_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
