import os
import pandas as pd
from deepface import DeepFace
import numpy as np

RAW_DIR = "data/raw"
PROCESSED_FILE = "data/processed/faces_data.csv"
TEST_RAW_DIR = "data/test_raw"
TEST_PROCESSED_FILE = "data/processed/test_faces_data.csv"


def extract_embedding(img_path):
    """Extract 512D face embedding using DeepFace (ArcFace model with MTCNN backend)."""
    try:
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name="ArcFace",
            detector_backend="mtcnn",
            enforce_detection=False
        )
        if embedding_objs and len(embedding_objs) > 0:
            return np.array(embedding_objs[0]["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"⚠️ Error processing {img_path}: {e}")
    return None


def process_dataset(raw_dir, output_file):
    """Process all face images in a folder structure and save embeddings to CSV."""
    embeddings_list = []
    labels_list = []

    for person_name in os.listdir(raw_dir):
        person_dir = os.path.join(raw_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"📷 Processing {person_name}...")
        for filename in os.listdir(person_dir):
            img_path = os.path.join(person_dir, filename)
            embedding = extract_embedding(img_path)
            if embedding is not None:
                # Normalize embedding for stability
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                embeddings_list.append(embedding)
                labels_list.append(person_name)

    # Convert lists to NumPy arrays
    if len(embeddings_list) == 0:
        print(f"⚠️ No valid embeddings found in {raw_dir}. Skipping.")
        return

    embeddings = np.vstack(embeddings_list)
    labels = np.array(labels_list, dtype=str)

    # Combine into a DataFrame
    df = pd.DataFrame(embeddings)
    df["label"] = labels

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df)} embeddings → {output_file}")


def main():
    print("🚀 Starting embedding extraction...")

    # Process training data
    process_dataset(RAW_DIR, PROCESSED_FILE)

    # Process testing data
    process_dataset(TEST_RAW_DIR, TEST_PROCESSED_FILE)

    print("🎉 Embedding extraction complete!")


if __name__ == "__main__":
    main()
