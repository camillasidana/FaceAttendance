import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
import os

DATA_CSV = "data/processed/faces_data.csv"

def main():
    if not os.path.exists(DATA_CSV):
        raise SystemExit("Run: python main.py --stage prep  (to create faces_data.csv)")
    df = pd.read_csv(DATA_CSV)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "logreg": make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=200)),
        "svm": make_pipeline(StandardScaler(with_mean=False), SVC(probability=True)),
        "dt": DecisionTreeClassifier()
    }
    os.makedirs("models", exist_ok=True)
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        dump(mdl, f"models/{name}.joblib")
        print(f"Saved models/{name}.joblib")

if __name__ == "__main__":
    main()
