import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print("✅ Camera found at index:", i)
        cap.release()
    else:
        print("❌ No camera at index:", i)
