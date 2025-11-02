import cv2
import numpy as np
from joblib import load
import json
from datetime import datetime, date
from deepface import DeepFace
import os
import sqlite3

# ---------- CONFIG ----------
BEST_MODEL_PATH = "models/best_model.json"
SCALER_PATH = "models/scaler.joblib"
PCA_PATH = "models/pca.joblib"  # optional, if PCA was used
CONFIDENCE_THRESHOLD = 0.98
DETECTOR = "mtcnn"
EMBEDDING_MODEL = "ArcFace"
FRAME_SKIP = 1
RESOLUTION = (640, 480)
DB_PATH = "data/attendance.db"  # 🧠 SQLite database file path


# ---------- DATABASE HELPERS ----------
def init_db():
    """Create the attendance database and table if not exist."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            status TEXT DEFAULT 'present',
            date TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print("🗄️ Database initialized at:", DB_PATH)


def mark_attendance_db(name, status="present"):
    """Insert attendance into the database; prevent duplicates per day."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    ts = datetime.now().isoformat(timespec='seconds')
    today = date.today().isoformat()

    # Check if already marked
    cur.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, today))
    existing = cur.fetchone()

    if existing:
        print(f"⚠️ {name} already marked today.")
    else:
        cur.execute("INSERT INTO attendance (name, timestamp, status, date) VALUES (?, ?, ?, ?)",
                    (name, ts, status, today))
        conn.commit()
        print(f"📋 Marked attendance for {name} at {ts}")

    conn.close()


def fetch_today():
    """Fetch today's attendance for display or logic."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    today = date.today().isoformat()
    cur.execute("SELECT name FROM attendance WHERE date=?", (today,))
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return set(rows)


# ---------- ML HELPERS ----------
def get_embedding(face_img):
    """Extract DeepFace embedding."""
    try:
        if face_img is None or face_img.size == 0:
            return None
        h, w, _ = face_img.shape
        if h < 40 or w < 40:
            return None

        cv2.imwrite("temp_face.jpg", face_img)
        emb = DeepFace.represent(
            img_path="temp_face.jpg",
            model_name=EMBEDDING_MODEL,
            detector_backend=DETECTOR,
            enforce_detection=False
        )
        if not emb:
            return None

        vec = np.array(emb[0]["embedding"], dtype=np.float32)
        if vec.size == 0:
            return None
        return vec
    except Exception as e:
        print(f"⚠️ Embedding error: {e}")
        return None


def identify_person_model(embedding, model, scaler, pca):
    """Predict identity using trained ML model."""
    try:
        embedding_scaled = scaler.transform([embedding])

        pred = model.predict(embedding_scaled)[0]

        # Compute model confidence if available
        if hasattr(model, "predict_proba"):
            conf = np.max(model.predict_proba(embedding_scaled))
            if conf < CONFIDENCE_THRESHOLD:
                return "Unknown", conf
            return pred, conf
        else:
            return pred, 1.0
    except Exception as e:
        print(f"⚠️ Model identification error: {e}")
        return "Unknown", 0.0


# ---------- MAIN ----------
def main():
    print("📦 Loading ML model...")

    # Initialize DB
    init_db()

    # Load best model info
    with open(BEST_MODEL_PATH, "r") as f:
        best_info = json.load(f)
    best_model_name = best_info["best_model"]
    print(f"🏆 Best model detected: {best_model_name.upper()}")

    # Load scaler and PCA (if present)
    scaler = load(SCALER_PATH)
    pca = load(PCA_PATH) if os.path.exists(PCA_PATH) else None

    # Load the trained ML model
    model_path = f"models/{best_model_name}.joblib"
    model = load(model_path)
    print(f"✅ Loaded {best_model_name.upper()} model")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    if not cap.isOpened():
        raise SystemExit("❌ Webcam not found.")

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("🎥 Camera started — press 'q' to quit.")
    frame_count = 0
    marked_today = fetch_today()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, RESOLUTION)
        frame_count += 1

        if frame_count % FRAME_SKIP != 0:
            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            if w < 40 or h < 40:
                continue

            roi = frame[y:y+h, x:x+w]
            embedding = get_embedding(roi)
            if embedding is None:
                continue

            name, conf = identify_person_model(embedding, model, scaler, pca)
            conf_text = f"conf={conf:.2f}"

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            if name != "Unknown":
                if name not in marked_today:
                    mark_attendance_db(name)
                    marked_today.add(name)
                    text = f"{name} ✅"
                else:
                    text = f"{name} (Already marked)"
            else:
                text = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{text} [{conf_text}]",
                        (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Attendance session ended.")
    print(f"✅ Attendance saved for: {', '.join(marked_today) if marked_today else 'No one'}")


if __name__ == "__main__":
    main()
