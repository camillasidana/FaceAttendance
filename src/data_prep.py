import os
import pandas as pd
from deepface import DeepFace
import numpy as np

RAW_DIR = "data/raw"
PROCESSED_FILE = "data/processed/faces_data.csv"
TEST_RAW_DIR = "data/test_raw"
TEST_PROCESSED_FILE = "data/processed/test_faces_data.csv"

def extract_embedding(img_path):
    """Extract 128–512D face embedding using DeepFace (Facenet model with PyTorch backend)."""
    try:
        embedding_objs = DeepFace.represent(
        img_path=img_path,
        model_name="ArcFace",
        detector_backend="mtcnn",
        enforce_detection=False
)

        if embedding_objs and len(embedding_objs) > 0:
            return embedding_objs[0]["embedding"]
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
    return None

def main():
    data = []
    labels = []

    for person_name in os.listdir(RAW_DIR):
        person_dir = os.path.join(RAW_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"Processing {person_name}...")
        for filename in os.listdir(person_dir):
            img_path = os.path.join(person_dir, filename)
            embedding = extract_embedding(img_path)
            embedding = embedding / np.linalg.norm(embedding)
            if embedding is not None:
                data.append(embedding)
                labels.append(person_name)

    # Save to CSV
    df1 = pd.DataFrame(data)
    df1["label"] = labels
    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
    df1.to_csv(PROCESSED_FILE, index=False)
    print(f"✅ Saved {len(df1)} embeddings → {PROCESSED_FILE}")

    data = []
    labels = []

    for person_name in os.listdir(TEST_RAW_DIR):
        person_dir = os.path.join(TEST_RAW_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"Processing {person_name}...")
        for filename in os.listdir(person_dir):
            img_path = os.path.join(person_dir, filename)
            embedding = extract_embedding(img_path)
            embedding = embedding / np.linalg.norm(embedding)
            if embedding is not None:
                data.append(embedding)
                labels.append(person_name)

    # Save to CSV
    df2 = pd.DataFrame(data)
    df2["label"] = labels
    os.makedirs(os.path.dirname(TEST_PROCESSED_FILE), exist_ok=True)
    df2.to_csv(TEST_PROCESSED_FILE, index=False)
    print(f"✅ Saved {len(df2)} embeddings → {TEST_PROCESSED_FILE}")

if __name__ == "__main__":
    main()
