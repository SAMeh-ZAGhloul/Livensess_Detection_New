# Liveness Detection Web App

A lightweight web application for facial liveness detection based on the [eKYC project](https://github.com/manhcuong02/eKYC/tree/main/liveness_detection). This application provides a simple interface for performing liveness detection as part of identity verification processes.

## Features

- Responsive web interface that works on both desktop and mobile devices
- Real-time webcam access for face capture
- Four-step liveness detection challenge sequence:
  1. Look directly at the camera (baseline position)
  2. Turn face right
  3. Turn face left
  4. Blink eyes
- **Visual feedback features:**
  - Clean, minimalist UI with no facial landmarks display
  - Step-by-step progress indicators
  - Fully automatic progression between steps
  - Visual feedback messages for each action
  - Final results display after completing all challenges
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
Note: Since the application uses a self-signed certificate, you may need to accept the security warning in your browser.

3. If you haven't downloaded the model yet (uncompressed model should placed in local fodler "model"), visit:
```
https://localhost:5555/download-model
```
Or from "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2"
        
4. Return to the main page and grant camera permissions when prompted by your browser.

5. Follow the on-screen instructions to complete the liveness detection process:
   - **Step 1**: Look directly at the camera - Center your face in the frame
   - **Step 2**: Turn your face right - Make a clear, deliberate movement to the right
   - **Step 3**: Turn your face left - Make a clear, deliberate movement to the left
   - **Step 4**: Blink your eyes - Close and open your eyes completely once or twice
   
   **Important blink detection tips:**
   - Close your eyes completely for a brief moment (about half a second)
   - Open your eyes again normally
   - The system requires 2 consecutive frames with closed eyes for detection
   - Ensure good lighting conditions for better detection

6. Each successful action will automatically advance to the next step. After completing all challenges, a final results summary is displayed.

## Project Structure

This project consists of only three files:

- `index.html`: The frontend interface with HTML, CSS, and JavaScript
- `app.py`: The Flask backend server that implements liveness detection using dlib and OpenCV
- `README.md`: This documentation file

Additionally, the application creates two directories:
- `model/`: Contains the facial landmark detection model file
- `cert/`: Contains SSL certificates for HTTPS

## How It Works

1. The frontend captures frames from the user's webcam at 2 FPS (frame per second).
2. Each frame is sent to the backend API for real-time processing.
3. The backend uses:
   - dlib's face detection to locate faces in the image
   - Face orientation detection with mirroring correction
   - Eye aspect ratio analysis for blink detection
   - Background thread processing for blink detection
4. When a challenge is successfully completed, detection stops and the app automatically advances to the next step.
5. After all challenges are completed, detection stops completely and a final verification summary is displayed.

## Blink Detection Parameters

The blink detection algorithm uses the following parameters:

1. **Eye Aspect Ratio (EAR) Threshold: 0.25**
   - When the eye aspect ratio falls below this value, eyes are considered closed
   - Higher values make detection more sensitive (detects smaller eye closures)
   - Lower values require more complete eye closure for detection

2. **Consecutive Frames: 2**
   - Requires eyes to be closed for 2 consecutive frames to detect a blink
   - At 1 FPS, this means eyes should be closed for approximately 1-2 seconds
   - Helps prevent false positives from momentary eye movements

3. **Dynamic Thresholding**
   - Calculates a baseline from recent eye aspect ratios
   - Adapts to different users' eye shapes and sizes
   - Improves reliability across different lighting conditions

## Security Features

The application includes several security enhancements:

1. **HTTPS Support**
   - Automatically generates self-signed SSL certificates in the `cert/` directory if not present
   - Serves all content over HTTPS for secure communication
   - Protects webcam data during transmission

2. **CORS Headers**
   - Implements Cross-Origin Resource Sharing (CORS) headers
   - Allows cross-origin requests from any domain
   - Supports OPTIONS preflight requests for CORS compatibility

## API Endpoints

- `/`: Serves the main application
- `/api/process-frame`: Processes individual frames for real-time feedback
- `/api/liveness-detection`: Processes the final verification with all captured frames
- `/download-model`: Downloads the required facial landmark model
- `/success`: Success page shown after successful verification

## License
MIT

## Acknowledgements
This project was inspired by and uses code from the [eKYC project](https://github.com/manhcuong02/eKYC) by manhcuong02, specifically the liveness_detection module.
