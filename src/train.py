import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib, json

# ---------- PATHS ----------
TRAIN_FILE = "data/processed/faces_data.csv"
EXTERNAL_TEST_FILE = "data/processed/test_faces_data.csv"
MODELS_DIR = "models"


def evaluate_model(name, model, X_test, y_test):
    """Evaluate a model and print accuracy, confusion matrix, and report."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ {name} accuracy: {acc:.3f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3))
    return acc


def main():
    # ---------- Load Data ----------
    if not os.path.exists(TRAIN_FILE):
        raise SystemExit("❌ faces_data.csv not found in processed folder.")

    df = pd.read_csv(TRAIN_FILE)
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    print("⚙️ Scaling features with StandardScaler ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.joblib")
    print("💾 Saved → models/scaler.joblib")

    # ---------- Train/Test Split ----------
    X_train, X_split_test, y_train, y_split_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------- Define Models ----------
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "svm": SVC(probability=True, kernel="rbf", C=1),
        "dt": DecisionTreeClassifier(random_state=42)
    }

    results_split = {}
    results_external = {}

    # ---------- Train and Evaluate on Split ----------
    print("\n=== TRAINING & SPLIT TEST EVALUATION ===")
    for name, model in models.items():
        print(f"\n🧠 Training {name.upper()} ...")
        model.fit(X_train, y_train)
        acc = evaluate_model(name, model, X_split_test, y_split_test)
        results_split[name] = acc
        joblib.dump(model, f"{MODELS_DIR}/{name}.joblib")
        print(f"💾 Saved → models/{name}.joblib")

    # ---------- Evaluate on External Test Dataset ----------
    if os.path.exists(EXTERNAL_TEST_FILE):
        print("\n=== EVALUATION ON EXTERNAL TEST DATA ===")
        df_test = pd.read_csv(EXTERNAL_TEST_FILE)
        X_test_ext = df_test.drop(columns=["label"]).values
        y_test_ext = df_test["label"].values
        X_test_ext_scaled = scaler.transform(X_test_ext)

        for name, model in models.items():
            print(f"\n📊 Testing {name.upper()} on external dataset ...")
            acc = evaluate_model(name, model, X_test_ext_scaled, y_test_ext)
            results_external[name] = acc
    else:
        print("⚠️ No external test dataset found. Skipping external evaluation.")
        results_external = {name: None for name in models.keys()}

    # ---------- Summary Table ----------
    print("\n📊 FINAL COMPARISON:")
    print(f"{'MODEL':<10} | {'Split Acc':<10} | {'External Acc':<10}")
    print("-" * 35)
    for name in results_split.keys():
        split_acc = results_split.get(name, 0)
        ext_acc = results_external.get(name, None)
        print(f"{name:<10} | {split_acc:<10.3f} | {ext_acc if ext_acc else '-':<10}")

    # ---------- Best Model Selection ----------
    all_results = {
        name: (results_external.get(name) or results_split.get(name, 0))
        for name in models.keys()
    }
    best_name = max(all_results, key=all_results.get)
    best_acc = all_results[best_name]

    print(f"\n🏆 Best model selected: {best_name.upper()} ({best_acc:.3f})")

    with open(f"{MODELS_DIR}/best_model.json", "w") as f:
        json.dump({"best_model": best_name, "accuracy": best_acc}, f, indent=4)

    print("\n🎉 Training and evaluation complete!")


if __name__ == "__main__":
    main()
