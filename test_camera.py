import cv2
import time

def test_camera(device_id):
    print(f"\nTrying device /dev/video{device_id}")
    # Set specific parameters for Raspberry Pi Camera
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {device_id}")
        return False
    
    # Give the camera time to warm up
    time.sleep(2)
    
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

# Test the unicam devices (Raspberry Pi Camera)
for device_id in [0, 1]:
    if test_camera(device_id):
        print(f"Found working camera at /dev/video{device_id}")
        break 