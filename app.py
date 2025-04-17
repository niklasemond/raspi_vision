import cv2
import numpy as np
from flask import Flask, render_template, Response
from ultralytics import YOLO
import threading
import time
import torch

app = Flask(__name__)

# Initialize YOLO model with explicit device
device = 'cpu'  # Force CPU usage
model = YOLO('yolov8n.pt').to(device)

# Global variables for video capture and frame processing
camera = cv2.VideoCapture(0)
frame_count = 0
detection_interval = 5  # Process every 5th frame to reduce CPU load
last_frame = None
last_detections = None
lock = threading.Lock()

def process_frame(frame):
    """Process a single frame with YOLO detection"""
    results = model(frame, conf=0.5, device=device)
    annotated_frame = results[0].plot()
    return annotated_frame

def generate_frames():
    """Generate video frames with object detection"""
    global frame_count, last_frame, last_detections
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        with lock:
            frame_count += 1
            
            # Only run detection every few frames
            if frame_count % detection_interval == 0:
                last_frame = frame.copy()
                last_detections = process_frame(frame)
            
            # Use the last processed frame if available
            if last_detections is not None:
                frame = last_detections
        
        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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
    app.run(host='0.0.0.0', port=5000, debug=False) 