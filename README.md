# Raspberry Pi Object Detection

A Flask application that performs real-time object detection using YOLOv8 on a Raspberry Pi with a camera module or USB webcam.

## Requirements

- Raspberry Pi 4 with 64-bit Raspberry Pi OS
- Raspberry Pi Camera Module or USB webcam
- Python 3.9 or higher

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. The YOLOv8n model will be automatically downloaded on first run.

## Usage

1. Connect your camera (either Raspberry Pi Camera Module or USB webcam)

2. Run the Flask application:
```bash
python app.py
```

3. Open a web browser and navigate to:
```
http://<raspberry-pi-ip>:5000
```

## Features

- Real-time object detection using YOLOv8n
- Optimized for Raspberry Pi performance
- Responsive web interface
- Automatic model download
- Configurable detection interval

## Performance Optimization

The application includes several optimizations for Raspberry Pi:
- Processes every 5th frame by default to reduce CPU load
- Uses threading to handle video capture and processing
- Implements frame buffering to maintain smooth video feed

## Configuration

You can adjust the following parameters in `app.py`:
- `detection_interval`: Change the number of frames between detections
- `conf`: Adjust the confidence threshold for detections

## Troubleshooting

If you encounter issues:
1. Ensure your camera is properly connected and accessible
2. Check if the camera is being used by another process
3. Verify all dependencies are installed correctly
4. Try reducing the detection interval if performance is poor 