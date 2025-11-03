import os
import pandas as pd
import numpy as np
from deepface import DeepFace
from sklearn.decomposition import PCA
import joblib

RAW_DIR = "data/raw"
PROCESSED_FILE = "data/processed/faces_data.csv"
TEST_RAW_DIR = "data/test_raw"
TEST_PROCESSED_FILE = "data/processed/test_faces_data.csv"
PCA_PATH = "models/pca.joblib"

def extract_embedding(img_path):
    """Extract 512D face embedding using DeepFace ArcFace + MTCNN."""
    try:
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name="ArcFace",
            detector_backend="mtcnn",
            enforce_detection=False
        )
        if embedding_objs and len(embedding_objs) > 0:
            return np.array(embedding_objs[0]["embedding"])
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
    return None


def process_folder(folder_path):
    """Return embeddings and labels for all images in a folder."""
    data, labels = [], []
    for person_name in os.listdir(folder_path):
        person_dir = os.path.join(folder_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"Processing {person_name}...")
        for filename in os.listdir(person_dir):
            img_path = os.path.join(person_dir, filename)
            emb = extract_embedding(img_path)
            if emb is not None:
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                data.append(emb)
                labels.append(person_name)

    return np.vstack(data), np.array(labels, dtype=str)


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("📦 Extracting training embeddings...")
    X_train, y_train = process_folder(RAW_DIR)
    df_train = pd.DataFrame(X_train)
    df_train["label"] = y_train
    df_train.to_csv(PROCESSED_FILE, index=False)
    print(f"✅ Saved {len(df_train)} samples to {PROCESSED_FILE}")

    print("📦 Extracting test embeddings...")
    X_test, y_test = process_folder(TEST_RAW_DIR)
    df_test = pd.DataFrame(X_test)
    df_test["label"] = y_test
    df_test.to_csv(TEST_PROCESSED_FILE, index=False)
    print(f"✅ Saved {len(df_test)} samples to {TEST_PROCESSED_FILE}")

    # ---- PCA INTEGRATION ----
    print("⚙️ Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=0.95, random_state=42)  # keep 95% variance
    pca.fit(X_train)
    joblib.dump(pca, PCA_PATH)
    print(f"💾 PCA model saved to {PCA_PATH}")

    reduced_dim = pca.transform(X_train).shape[1]
    print(f"✅ Reduced dimensionality: {X_train.shape[1]} → {reduced_dim}")

if __name__ == "__main__":
    main()
