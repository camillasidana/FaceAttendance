import os
import pandas as pd
from deepface import DeepFace

RAW_DIR = "data/raw"
PROCESSED_FILE = "data/processed/faces_data.csv"

def extract_embedding(img_path):
    """Extract 128–512D face embedding using DeepFace (Facenet model with PyTorch backend)."""
    try:
        embedding_objs = DeepFace.represent(
        img_path=img_path,
        model_name="SFace",
        detector_backend="opencv",
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
            if embedding is not None:
                data.append(embedding)
                labels.append(person_name)

    # Save to CSV
    df = pd.DataFrame(data)
    df["label"] = labels
    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"✅ Saved {len(df)} embeddings → {PROCESSED_FILE}")

if __name__ == "__main__":
    main()
