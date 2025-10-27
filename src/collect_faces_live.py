import cv2
import os
from datetime import datetime


def collect_face_data():
    # Ask for person's name
    person_name = input("Enter the person's name: ").strip()

    if not person_name:
        print("Name cannot be empty!")
        return

    # Create folder for this person in known_faces
    person_folder = os.path.join('known_faces', person_name)
    os.makedirs(person_folder, exist_ok=True)

    # Choose camera (0 for laptop, 1 for DroidCam - adjust as needed)
    camera_index = int(input("Enter camera index (0 for laptop, 1 for DroidCam): "))

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}")
        return

    print(f"\n📸 Collecting photos for: {person_name}")
    print("Instructions:")
    print("- Press SPACE to capture a photo")
    print("- Press Q to quit")
    print("- Capture 20-30 photos from different angles and expressions")
    print("- Move your face: left, right, up, down, smile, neutral, etc.\n")

    photo_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Display instructions on frame
        cv2.putText(frame, f"Person: {person_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Photos captured: {photo_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: Capture | Q: Quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Collect Face Data', frame)

        key = cv2.waitKey(1) & 0xFF

        # Press SPACE to capture
        if key == ord(' '):
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{person_name}_{photo_count}_{timestamp}.jpg"
            filepath = os.path.join(person_folder, filename)

            cv2.imwrite(filepath, frame)
            print(f"✓ Captured: {filename}")

        # Press Q to quit
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Total photos collected: {photo_count}")
    print(f"📁 Saved in: {person_folder}")


if __name__ == "__main__":
    collect_face_data()