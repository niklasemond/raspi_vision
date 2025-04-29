import cv2
import numpy as np
from flask import Flask, render_template, Response
from ultralytics import YOLO
import threading
import time
import torch
from picamera2 import Picamera2
from queue import Queue
import threading
import os
import signal
import sys
import libcamera

app = Flask(__name__)

# Initialize YOLO model with optimized settings
device = 'cpu'  # Force CPU usage
model = YOLO('yolov8n.pt').to(device)

# Global variables
picam2 = None
frame_queue = Queue(maxsize=1)  # Single frame queue to prevent lag
detection_interval = 45  # Process every 45th frame
last_detections = None
lock = threading.Lock()
processing_resolution = (160, 128)  # Further reduced resolution for faster processing

def init_camera():
    """Initialize camera with retry mechanism"""
    global picam2
    
    # Try to kill any existing processes using the camera
    os.system('sudo pkill -f libcamera')
    time.sleep(1)  # Wait for processes to terminate
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            picam2 = Picamera2()
            # Configure camera with maximum field of view
            config = picam2.create_preview_configuration(
                main={"size": (1640, 1232), "format": "RGB888"},  # Use sensor's native resolution
                transform=libcamera.Transform(hflip=0, vflip=0),  # No flip
                buffer_count=2  # Reduced buffer count for lower latency
            )
            # Set controls for maximum field of view
            controls = {
                "ScalerCrop": (0, 0, 1640, 1232),  # Full sensor area
                "AeEnable": True,
                "AwbEnable": True,
                "FrameDurationLimits": (33333, 33333)  # 30 fps
            }
            picam2.configure(config)
            picam2.set_controls(controls)
            picam2.start()
            print(f"Camera initialized successfully on attempt {attempt + 1}")
            return True
        except Exception as e:
            print(f"Camera initialization attempt {attempt + 1} failed: {e}")
            if picam2 is not None:
                try:
                    picam2.close()
                except:
                    pass
            time.sleep(1)  # Wait before retrying
    
    print("Failed to initialize camera after multiple attempts")
    return False

def cleanup(signum, frame):
    """Cleanup function to be called on exit"""
    global picam2
    if picam2 is not None:
        try:
            picam2.stop()
            picam2.close()
        except:
            pass
    sys.exit(0)

# Register cleanup handler
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# Initialize camera
if not init_camera():
    print("Exiting due to camera initialization failure")
    sys.exit(1)

# Give the camera time to warm up
time.sleep(2)

def detection_thread():
    """Separate thread for object detection"""
    global last_detections
    frame_count = 0
    
    while True:
        try:
            frame = frame_queue.get()
            frame_count += 1
            
            if frame_count % detection_interval == 0:
                # Resize frame to smaller size for faster processing
                small_frame = cv2.resize(frame, processing_resolution)
                
                # Process frame with optimized settings
                results = model(small_frame, 
                              conf=0.4,  # Slightly lower confidence threshold
                              device=device,
                              imgsz=processing_resolution)
                
                # Resize detections back to original size
                with lock:
                    if results[0].boxes is not None:
                        # Create blank image at original size
                        blank = np.zeros((480, 640, 3), dtype=np.uint8)
                        # Resize the annotated frame
                        resized = cv2.resize(results[0].plot(), (640, 480))
                        # Overlay detections on original frame with higher opacity
                        last_detections = cv2.addWeighted(frame, 0.8, resized, 0.2, 0)
                    else:
                        last_detections = frame
            
            frame_queue.task_done()
        except Exception as e:
            print(f"Error in detection thread: {e}")
            break

def generate_frames():
    """Generate video frames with object detection"""
    global last_detections
    
    # Start detection thread
    detector = threading.Thread(target=detection_thread, daemon=True)
    detector.start()
    
    frame_count = 0
    while True:
        try:
            # Skip frames if queue is not empty to prevent lag
            if frame_queue.empty():
                # Capture frame from picamera2
                frame = picam2.capture_array()
                
                # Convert from RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Resize frame to display size while maintaining aspect ratio
                frame = cv2.resize(frame, (640, 480))
                
                # Add frame to queue
                frame_queue.put(frame.copy())
            
            # Use the last processed frame if available, otherwise use current frame
            with lock:
                if last_detections is not None:
                    display_frame = last_detections
                else:
                    display_frame = frame
            
            # Encode the frame with optimized settings
            ret, buffer = cv2.imencode('.jpg', display_frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Smaller delay to improve responsiveness
            time.sleep(0.005)
            
        except Exception as e:
            print(f"Error in frame generation: {e}")
            break

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        cleanup(None, None) 