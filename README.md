# Liveness Detection Web App

A lightweight web application for facial liveness detection based on the [eKYC project](https://github.com/manhcuong02/eKYC/tree/main/liveness_detection). This application provides a simple interface for performing liveness detection as part of identity verification processes.

## Features
- Responsive web interface that works on both desktop and mobile devices
- Real-time webcam access for face capture
- Four-step liveness detection challenge sequence:
  1. Turn face right
  2. Turn face left
  3. Blink eyes

- **Security features:**
  - HTTPS support with self-signed certificates
  - CORS headers for cross-origin requests

## Requirements
- Python 3.10 or higher
- Flask
- OpenCV (cv2)
- dlib
- NumPy
- imutils
- PIL (Pillow)
- torch (PyTorch)
- pyOpenSSL (for HTTPS certificate generation)
- Modern web browser with webcam access

## Installation
1. Clone this repository:
```bash
git clone https://github.com/SAMeh-ZAGhloul/Livensess_Detection_New.git
cd liveness-detection-app
```
2. Install the required Python dependencies:
```bash
pip install flask opencv-python dlib numpy imutils pillow torch pyopenssl
```

3. Create necessary directories:
```bash
mkdir -p model cert
```

Note: Installing dlib might require additional system dependencies. On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
```

## Download Required Model
Before running the application, you need to download the facial landmark detection model. You can do this in two ways:
1. **Automatic download**: Visit `/download-model` endpoint after starting the application
2. **Manual download**:
   - Download the model from [dlib-models](https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2)
   - Extract the .bz2 file
   - Place the extracted .dat file in the `model` directory with the name `shape_predictor_68_face_landmarks_GTX.dat`

## Usage
1. Start the Flask server:
```bash
python3 app.py
```
2. Open your web browser and navigate to:
```
https://localhost:5555
```

3. Downloaded the model (uncompressed model should placed in local fodler "model"):
```
https://localhost:5555/download-model
```
Or from "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2"
        
4. Return to the main page and grant camera permissions when prompted by your browser.

## Project Structure
This project consists of only three files:
- `index.html`: The frontend interface with HTML, CSS, and JavaScript
- `app.py`: The Flask backend server that implements liveness detection using dlib and OpenCV
- `README.md`: This documentation file

Additionally, the application creates two directories:
- `model/`: Contains the facial landmark detection model file
- `cert/`: Contains SSL certificates for HTTPS

## API Endpoints
- `/`: Serves the main application
- `/api/process-frame`: Processes individual frames for real-time feedback
- `/api/liveness-detection`: Processes the final verification with all captured frames
- `/download-model`: Downloads the required facial landmark model
- `/success`: Success page shown after successful verification
