import cv2
from joblib import load
import numpy as np
from datetime import datetime
from src.db import init_db, add_employee, mark_attendance

# Load Haar cascade for face detection
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Path to trained model
MODEL_PATH = "models/svm.joblib"  # Change if needed

def predict_face(model, face_img):
    """Resize and flatten the detected face for prediction."""
    face_resized = cv2.resize(face_img, (64, 64)).flatten().astype(np.float32).reshape(1, -1)
    return model.predict(face_resized)[0]

def main():
    """Main function for running live face recognition attendance."""
    init_db()  # Initialize database if not exists
    model = load(MODEL_PATH)

    # Open webcam (0 = default, use 1 if external camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Webcam not found.")
    
    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ✅ Rotate frame for correct orientation (for mobile webcam)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = CASCADE.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            name = predict_face(model, roi)

            # ✅ Ignore if it's a non-face detection
            if name.lower() != "non_face":
                ts = datetime.now().isoformat(timespec="seconds")
                add_employee(name)
                mark_attendance(name, ts, "present")
                color = (0, 255, 0)  # Green for valid face
            else:
                color = (0, 0, 255)  # Red for non-face

            # Draw box and label on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show the feed
        cv2.imshow("Attendance", frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
