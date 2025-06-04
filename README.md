# Liveness Detection Web App

A lightweight web application for facial liveness detection based on the [eKYC project](https://github.com/manhcuong02/eKYC/tree/main/liveness_detection). This application provides a simple interface for performing liveness detection as part of identity verification processes.

## Features

- Responsive web interface that works on both desktop and mobile devices
- Real-time webcam access for face capture
- Multiple liveness detection challenges:
  - Face orientation detection (turn face left/right)
  - Enhanced blink detection with improved sensitivity
- **Visual feedback features:**
  - Clean, minimalist UI with no facial landmarks display
  - Step-by-step progress indicators
  - Fully automatic progression between steps
  - Visual feedback messages for each action
  - Final results display after completing all challenges
- **Security features:**
  - HTTPS support with self-signed certificates
  - CORS headers for cross-origin requests
- Clean, modern UI with intuitive user experience

## Requirements

- Python 3.6 or higher
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
git clone https://github.com/yourusername/liveness-detection-app.git
cd liveness-detection-app
```

2. Install the required Python dependencies:
```bash
pip install flask opencv-python dlib numpy imutils pillow torch pyopenssl
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
   - Place the extracted .dat file in the application directory with the name `shape_predictor_68_face_landmarks_GTX.dat`

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
https://localhost:5555
```
Note: Since the application uses a self-signed certificate, you may need to accept the security warning in your browser.

3. If you haven't downloaded the model yet, visit:
```
https://localhost:5555/download-model
```

4. Return to the main page and grant camera permissions when prompted by your browser.

5. Follow the on-screen instructions to complete the liveness detection process:
   - **Step 1**: Turn your face right - Make a clear, deliberate movement to the right
   - **Step 2**: Turn your face left - Make a clear, deliberate movement to the left
   - **Step 3**: Blink your eyes - A single clear blink is sufficient (close and open your eyes completely)
   
   **Important blink detection tips:**
   - Close your eyes completely for a brief moment (about half a second)
   - Open your eyes again normally
   - Only one clear blink is required for detection
   - If detection fails after several attempts, try a slightly longer eye closure
   - Ensure good lighting conditions for better detection

6. Each successful action will automatically advance to the next step. After completing all challenges, a final results summary is displayed.

## Project Structure

This project consists of only three files:

- `index.html`: The frontend interface with HTML, CSS, and JavaScript
- `app.py`: The Flask backend server that implements liveness detection using dlib and OpenCV
- `README.md`: This documentation file

## How It Works

1. The frontend captures frames from the user's webcam at 1 FPS (frame per second).
2. Each frame is sent to the backend API for real-time processing.
3. The backend uses:
   - dlib's face detection to locate faces in the image
   - Enhanced face orientation detection with mirroring correction
   - Improved eye aspect ratio analysis for more reliable blink detection
   - Background thread processing for blink detection
4. When a challenge is successfully completed, detection stops and the app automatically advances to the next step.
5. After all challenges are completed, detection stops completely and a final verification summary is displayed.

## Blink Detection Improvements

The blink detection algorithm has been significantly enhanced:

1. **Dynamic thresholding** - Calculates a baseline from recent eye aspect ratios and detects blinks when the ratio drops below 70% of the baseline
2. **Personalized detection** - Tracks minimum eye aspect ratio seen for each user to create a personalized threshold
3. **Background processing** - Uses a separate thread to analyze frames for blink detection, improving responsiveness
4. **Frame queueing** - Maintains a queue of recent frames to ensure no blinks are missed
5. **Lowered threshold** - Adjusted the eye aspect ratio threshold from 0.25 to 0.15 for better sensitivity
6. **Reduced consecutive frames** - Only requires 1 frame below threshold to detect a blink
7. **Memory management** - Improved memory handling to prevent crashes and ensure stability
8. **Detailed logging** - Added comprehensive logging for troubleshooting blink detection issues

## Security Features

The application includes several security enhancements:

1. **HTTPS Support**
   - Automatically generates self-signed SSL certificates if not present
   - Serves all content over HTTPS for secure communication
   - Protects webcam data during transmission

2. **CORS Headers**
   - Implements Cross-Origin Resource Sharing (CORS) headers
   - Allows cross-origin requests from any domain
   - Supports OPTIONS preflight requests for CORS compatibility

## Communication Protocol Recommendations

Based on our research, here are recommendations for improving the communication protocol:

1. **WebSockets** (Recommended for this application)
   - Provides full-duplex, bidirectional communication
   - Lower latency than HTTP for real-time applications
   - Native browser support
   - Ideal for continuous streaming of webcam frames
   - Better for real-time feedback and immediate server responses

2. **gRPC** (Alternative for non-browser applications)
   - Uses HTTP/2 for more efficient binary communication
   - Strongly typed with protocol buffers
   - Excellent for multi-language environments
   - Limited browser support (requires gRPC-Web proxy)
   - Better for structured, RPC-style communication

3. **HTTP/3** (Not HTTP/1.3)
   - Note: HTTP/1.3 doesn't exist; HTTP/3 is the latest version
   - HTTP/3 uses QUIC protocol instead of TCP
   - Improved performance over HTTP/1.1 but still request-response based
   - Not ideal for continuous bidirectional communication
   - Would require polling or long-polling techniques

For this liveness detection application, implementing WebSockets would provide the most significant performance improvement, enabling real-time bidirectional communication with lower latency and overhead compared to the current HTTPS approach.

## API Endpoints

- `/`: Serves the main application
- `/api/process-frame`: Processes individual frames for real-time feedback
- `/api/liveness-detection`: Processes the final verification with all captured frames
- `/download-model`: Downloads the required facial landmark model
- `/success`: Success page shown after successful verification
- `/api/protocol-info`: Returns information about available communication protocols

## Troubleshooting

- **Camera access denied**: Make sure to grant camera permissions in your browser
- **No face detected**: Ensure adequate lighting and that your face is clearly visible
- **Face orientation not detected correctly**: 
  - Make sure to turn your face clearly to the left or right
  - Ensure your face is well-lit and clearly visible
  - Try turning your face more distinctly in the requested direction
- **Blink not detected**: 
  - Try closing your eyes completely for about half a second
  - Make sure your eyes are clearly visible and not obscured
  - Ensure you're in a well-lit environment
  - Try a slightly longer eye closure if detection fails
- **HTTPS certificate warning**: This is expected with self-signed certificates; you can safely proceed
- **Model not found**: Visit the `/download-model` endpoint to download the required model
- **Memory errors or crashes**: Restart the application; the improved memory management should prevent most issues
- **Installation issues with dlib**: Refer to [dlib installation guide](http://dlib.net/compile.html) for platform-specific instructions

## License

MIT

## Acknowledgements

This project was inspired by and uses code from the [eKYC project](https://github.com/manhcuong02/eKYC) by manhcuong02, specifically the liveness_detection module.
