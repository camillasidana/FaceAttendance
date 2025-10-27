import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import load
import os

DATA_CSV = "data/processed/faces_data.csv"

def main():
    if not os.path.exists(DATA_CSV):
        raise SystemExit("Run: python main.py --stage prep")
    df = pd.read_csv(DATA_CSV)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    for name in ["logreg", "svm", "dt"]:
        path = f"models/{name}.joblib"
        if not os.path.exists(path):
            print(f"Model not found: {path} (skipping)")
            continue
        mdl = load(path)
        yhat = mdl.predict(Xte)
        acc = accuracy_score(yte, yhat)
        print(f"\n== {name} ==")
        print("Accuracy:", round(acc, 4))
        print("Confusion matrix:\n", confusion_matrix(yte, yhat))
        print("Report:\n", classification_report(yte, yhat))

if __name__ == "__main__":
    main()
