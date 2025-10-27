import cv2
import numpy as np
from joblib import load
import json
from datetime import datetime
from deepface import DeepFace
from src.db import init_db, add_employee, mark_attendance
from scipy.spatial.distance import cosine

# ---------- CONFIG ----------
MODEL_PATH = "models/svm.joblib"
MEANS_PATH = "models/means.json"
THRESHOLD = 0.40       # Lower = stricter; higher = more tolerant
DETECTOR = "opencv" 
EMBEDDING_MODEL = "SFace"  # ⚡ Fast & accurate
FRAME_SKIP = 1    
RESOLUTION = (640, 480) # Webcam resolution
ROTATE_MODE = cv2.ROTATE_90_CLOCKWISE  # Fix camera orientation

# ---------- HELPERS ----------
def get_embedding(face_img):
    """Extract DeepFace embedding for a detected face."""
    try:
        cv2.imwrite("temp_face.jpg", face_img)
        emb = DeepFace.represent(
            img_path="temp_face.jpg",
            model_name=EMBEDDING_MODEL,
            detector_backend=DETECTOR,
            enforce_detection=False
        )
        return np.array(emb[0]["embedding"]) if emb else None
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def identify_person(embedding, means):
    """Compare embedding to stored mean embeddings; return best match or Unknown."""
    best_label, best_score = None, 1e9
    for name, mean_vec in means.items():
        score = cosine(embedding, mean_vec)
        if score < best_score:
            best_score, best_label = score, name
    return best_label if best_score < THRESHOLD else "Unknown"

# ---------- MAIN LOOP ----------
def main():
    init_db()

    print("📦 Loading models...")
    model = load(MODEL_PATH)
    with open(MEANS_PATH, "r") as f:
        means = json.load(f)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

    if not cap.isOpened():
        raise SystemExit("❌ Webcam not found.")

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("🎥 Camera started — press 'q' to quit.")
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, RESOLUTION)

        frame_count += 1

        # Skip frames to reduce lag
        if frame_count % FRAME_SKIP != 0:
            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            roi_color = frame[y:y+h, x:x+w]
            embedding = get_embedding(roi_color)

            if embedding is not None:
                name = identify_person(embedding, means)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                if name != "Unknown":
                    ts = datetime.now().isoformat(timespec="seconds")
                    add_employee(name)
                    mark_attendance(name, ts, "present")

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, name, (x, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Attendance session ended.")

if __name__ == "__main__":
    main()
