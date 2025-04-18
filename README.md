# Raspberry Pi Object Detection

A Flask application that performs real-time object detection using YOLOv8 on a Raspberry Pi with the Raspberry Pi Camera Module.

## Requirements

- Raspberry Pi 4 with 64-bit Raspberry Pi OS
- Raspberry Pi Camera Module (IMX219 sensor)
- Python 3.9 or higher

## Installation

1. Clone this repository:
```bash
git clone https://github.com/niklasemond/raspi_vision.git
cd raspi_vision
```

2. Install system dependencies:
```bash
sudo apt update
sudo apt install -y python3-libcamera python3-picamera2
```

3. Create and activate a virtual environment with system site packages:
```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
```

4. Install Python dependencies:
```bash
pip install ultralytics==8.0.196 opencv-python==4.9.0.80 flask==3.0.2 numpy==1.26.4 torch==2.0.1 torchvision==0.15.2
```

## Usage

1. Make sure the Raspberry Pi Camera Module is properly connected:
   - The ribbon cable should be inserted with the blue side facing the Ethernet port
   - The cable should be fully inserted and the locking mechanism engaged

2. Activate the virtual environment (if not already activated):
```bash
source venv/bin/activate
```

3. Run the Flask application:
```bash
python app.py
```

4. Access the web interface from any device on your local network:
```
http://<raspberry-pi-ip>:5000
```

To find your Raspberry Pi's IP address:
```bash
hostname -I
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
- Uses picamera2 for efficient camera access

## Configuration

You can adjust the following parameters in `app.py`:
- `detection_interval`: Change the number of frames between detections
- `conf`: Adjust the confidence threshold for detections

## Troubleshooting

If you encounter issues:
1. Ensure the camera is properly connected and the ribbon cable is inserted correctly
2. Verify the camera is enabled in Raspberry Pi settings
3. Check if the camera is detected: `libcamera-hello --list-cameras`
4. Try reducing the detection interval if performance is poor
5. Make sure you're using the virtual environment with system site packages

## Safe Shutdown

To safely shut down the Raspberry Pi:
1. Stop the Flask application (Ctrl+C)
2. Run: `sudo shutdown -h now`
3. Wait until all LEDs stop blinking before removing power

## License

This project is licensed under the MIT License - see the LICENSE file for details. 