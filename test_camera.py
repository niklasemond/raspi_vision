import cv2
import os

def test_camera(device_id):
    print(f"\nTrying device /dev/video{device_id}")
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {device_id}")
        return False
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame from camera {device_id}")
        cap.release()
        return False
    
    print(f"Successfully captured frame from camera {device_id}")
    print(f"Frame shape: {frame.shape}")
    
    # Release the camera
    cap.release()
    return True

# Test all video devices
for i in range(32):  # Try up to video31
    if os.path.exists(f"/dev/video{i}"):
        if test_camera(i):
            print(f"Found working camera at /dev/video{i}")
            break 