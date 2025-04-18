from picamera2 import Picamera2
import time

def test_camera():
    print("Initializing camera...")
    picam2 = Picamera2()
    
    # Configure camera
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    
    # Start camera
    picam2.start()
    
    # Give the camera time to warm up
    time.sleep(2)
    
    # Capture a frame
    try:
        frame = picam2.capture_array()
        print("Successfully captured frame")
        print(f"Frame shape: {frame.shape}")
        return True
    except Exception as e:
        print(f"Error capturing frame: {e}")
        return False
    finally:
        picam2.stop()

if __name__ == "__main__":
    test_camera() 