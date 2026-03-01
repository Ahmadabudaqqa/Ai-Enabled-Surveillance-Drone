import cv2

for i in range(19, 36):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera found at index {i}")
            cap.release()
            break
        cap.release()
else:
    print("No working camera found in indices 19-35.")
