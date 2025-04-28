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

app = Flask(__name__)

# Initialize YOLO model with explicit device
device = 'cpu'  # Force CPU usage
model = YOLO('yolov8n.pt').to(device)

# Initialize camera with optimized settings
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

# Give the camera time to warm up
time.sleep(2)

# Global variables for video capture and frame processing
frame_queue = Queue(maxsize=2)  # Small queue to prevent lag
detection_interval = 15  # Process every 15th frame
last_detections = None
lock = threading.Lock()

def detection_thread():
    """Separate thread for object detection"""
    global last_detections
    frame_count = 0
    
    while True:
        try:
            frame = frame_queue.get()
            frame_count += 1
            
            if frame_count % detection_interval == 0:
                # Process frame for detection
                results = model(frame, conf=0.5, device=device)
                with lock:
                    last_detections = results[0].plot()
            
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
    
    while True:
        try:
            # Capture frame from picamera2
            frame = picam2.capture_array()
            
            # Convert from RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add frame to queue if not full
            if not frame_queue.full():
                frame_queue.put(frame.copy())
            
            # Use the last processed frame if available, otherwise use current frame
            with lock:
                if last_detections is not None:
                    display_frame = last_detections
                else:
                    display_frame = frame
            
            # Encode the frame with optimized settings
            ret, buffer = cv2.imencode('.jpg', display_frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
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
        picam2.stop() 