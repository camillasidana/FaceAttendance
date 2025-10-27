import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib, json

PROCESSED_FILE = "data/processed/faces_data.csv"

def main():
    # ---------- Load prepared embeddings ----------
    if not os.path.exists(PROCESSED_FILE):
        raise SystemExit("❌ No processed data found. Run --stage prep first.")
    df = pd.read_csv(PROCESSED_FILE)

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # ---------- Split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ---------- Define models ----------
    models = {
        "logreg": LogisticRegression(max_iter=500),
        "svm": SVC(probability=True, kernel="rbf", C=1),
        "dt": DecisionTreeClassifier(random_state=42)
    }

    os.makedirs("models", exist_ok=True)

    # ---------- Train + evaluate ----------
    for name, model in models.items():
        print(f"\n🧠 Training {name.upper()} model ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"✅ {name} accuracy: {acc:.3f}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=3))

        joblib.dump(model, f"models/{name}.joblib")
        print(f"💾 Saved → models/{name}.joblib")

    # ---------- Save mean embeddings per person ----------
    means = {}
    for label in np.unique(y_train):
        means[label] = np.mean(X_train[y_train == label], axis=0).tolist()
    with open("models/means.json", "w") as f:
        json.dump(means, f)
    print("💾 Saved → models/means.json (for unknown-face thresholding)")

    print("\n🎉 Training complete!")

if __name__ == "__main__":
    main()

