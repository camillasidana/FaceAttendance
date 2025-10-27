import os
import cv2
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
OUT_CSV = "data/processed/faces_data.csv"

def iter_images():
    for person in os.listdir(RAW_DIR):
        pdir = os.path.join(RAW_DIR, person)
        if not os.path.isdir(pdir):
            continue
        for fn in os.listdir(pdir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                yield person, os.path.join(pdir, fn)

def image_to_vector(path, size=(64, 64)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, size)
    return img.flatten()

def main():
    rows = []
    for label, path in iter_images():
        vec = image_to_vector(path)
        if vec is None:
            continue
        rows.append(np.concatenate([vec, np.array([label])], axis=0))
    if not rows:
        print("No images found in data/raw/. Add images in subfolders first.")
        return
    arr = np.vstack(rows)
    n_features = arr.shape[1] - 1
    cols = [f"px{i}" for i in range(n_features)] + ["label"]
    df = pd.DataFrame(arr, columns=cols)
    df["label"] = df["label"].astype(str)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} samples → {OUT_CSV}")

if __name__ == "__main__":
    main()
