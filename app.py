from flask import Flask, request, jsonify, send_from_directory
import os
import base64
import numpy as np
import cv2
import dlib
import torch
from imutils import face_utils
import math
import io
from PIL import Image
import re
import time
import threading
import queue
import ssl
import logging
import gc
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Create necessary directories if they don't exist
os.makedirs('model', exist_ok=True)
os.makedirs('cert', exist_ok=True)

# Path to store the landmark model - using local path as requested
LANDMARK_PATH = 'model/shape_predictor_68_face_landmarks_GTX.dat'

# Queue for storing frames for blink detection
blink_frames_queue = queue.Queue(maxsize=10)
blink_result_lock = threading.Lock()
blink_result = {'detected': False, 'timestamp': 0}

# Configure CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Override Flask's jsonify to use our custom encoder
def custom_jsonify(*args, **kwargs):
    return app.response_class(
        json.dumps(dict(*args, **kwargs), cls=NumpyEncoder),
        mimetype='application/json'
    )

class BlinkDetector:
    '''A class for detecting eye blinking in facial images'''
    def __init__(self):
        # Load model for eye landmark detection
        self.predictor_eyes = dlib.shape_predictor(LANDMARK_PATH)
        # Using the requested threshold values
        self.EYE_AR_THRESH = 0.25  # Updated as requested
        self.EYE_AR_CONSEC_FRAMES = 2  # Updated as requested (need 2 frames for detection)
        self.counter = 0
        self.total = 0
        # Store recent EAR values for analysis and dynamic thresholding
        self.recent_ears = []
        self.max_recent = 15  # Store more EAR values for better baseline calculation
        self.baseline_ear = None
        self.min_ear_seen = 1.0  # Track minimum EAR value seen
        self.blink_detected_frames = []  # Store frames where blinks were detected

    def eye_blink(self, rgb_image, rect, thresh=1):
        ''' 
        Detects eye blinking in a given face region of an input RGB image.
        Parameters:
        - rgb_image (np.ndarray): Input RGB image as a numpy array.
        - rect: A bounding rectangle [x1, y1, x2, y2] defining the face region.
        - thresh (int): A challenge-response threshold that the user needs to surpass.
        Returns:
        - out (bool): True if the user successfully surpasses the challenge (>= thresh), False otherwise (< thresh).
        '''
        try:
            # Convert rect to dlib rectangle if needed
            if isinstance(rect, torch.Tensor):
                rect = dlib.rectangle(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
            elif isinstance(rect, (np.ndarray, list, tuple)):
                rect = np.array(rect).astype(np.int32)
                rect = dlib.rectangle(rect[0], rect[1], rect[2], rect[3])
            elif isinstance(rect, dlib.rectangle):
                # Already a dlib rectangle, no conversion needed
                pass
            else:
                # Unknown type, try to convert to dlib rectangle
                try:
                    rect = dlib.rectangle(int(rect.left()), int(rect.top()), 
                                         int(rect.right()), int(rect.bottom()))
                except:
                    logger.error(f"Failed to convert rect of type {type(rect)} to dlib rectangle")
                    return False
                
            # Make a copy of the image to avoid modifying the original
            gray = cv2.cvtColor(rgb_image.copy(), cv2.COLOR_RGB2GRAY)
            
            # Get facial landmarks indices
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            
            # Determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = self.predictor_eyes(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            
            # Average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            
            # Update minimum EAR seen
            if ear < self.min_ear_seen:
                self.min_ear_seen = ear
            
            # Add to recent EARs
            self.recent_ears.append(ear)
            if len(self.recent_ears) > self.max_recent:
                self.recent_ears.pop(0)
            
            # Calculate baseline (average of highest EARs)
            if len(self.recent_ears) >= 5:
                # Sort EARs and take average of top 5 values as baseline
                sorted_ears = sorted(self.recent_ears, reverse=True)
                self.baseline_ear = sum(sorted_ears[:5]) / 5
                
                # Dynamic threshold based on baseline and minimum seen
                # This creates a more personalized threshold for each user
                if self.baseline_ear is not None:
                    # If we've seen a significant drop in EAR, use that as reference
                    if self.min_ear_seen < self.baseline_ear * 0.7:
                        # Use 60% of the way from min to baseline as threshold
                        dynamic_threshold = self.min_ear_seen + (self.baseline_ear - self.min_ear_seen) * 0.6
                    else:
                        # Otherwise use 70% of baseline
                        dynamic_threshold = self.baseline_ear * 0.7
                    
                    # Use the dynamic threshold, but never go higher than our static threshold
                    effective_threshold = min(dynamic_threshold, self.EYE_AR_THRESH)
                    
                    # Check if current EAR is below our effective threshold
                    if ear < effective_threshold:
                        self.counter += 1
                        # Store this frame as a potential blink frame
                        self.blink_detected_frames.append(ear)
                        # Keep only the last 5 blink frames
                        if len(self.blink_detected_frames) > 5:
                            self.blink_detected_frames.pop(0)
                    else:
                        # If the eyes were closed for a sufficient number of frames,
                        # then increment the total number of blinks
                        if self.counter >= self.EYE_AR_CONSEC_FRAMES:
                            self.total += 1
                            logger.info(f"Blink detected! EAR: {ear}, Threshold: {effective_threshold}, Baseline: {self.baseline_ear}")
                        # Reset the eye frame counter
                        self.counter = 0
                else:
                    # Fallback to static threshold
                    if ear < self.EYE_AR_THRESH:
                        self.counter += 1
                        self.blink_detected_frames.append(ear)
                        if len(self.blink_detected_frames) > 5:
                            self.blink_detected_frames.pop(0)
                    else:
                        if self.counter >= self.EYE_AR_CONSEC_FRAMES:
                            self.total += 1
                            logger.info(f"Blink detected with static threshold! EAR: {ear}, Threshold: {self.EYE_AR_THRESH}")
                        self.counter = 0
            else:
                # Use static threshold until we have enough data
                if ear < self.EYE_AR_THRESH:
                    self.counter += 1
                    self.blink_detected_frames.append(ear)
                    if len(self.blink_detected_frames) > 5:
                        self.blink_detected_frames.pop(0)
                else:
                    if self.counter >= self.EYE_AR_CONSEC_FRAMES:
                        self.total += 1
                        logger.info(f"Blink detected with initial static threshold! EAR: {ear}, Threshold: {self.EYE_AR_THRESH}")
                    self.counter = 0
                
            # Only need 1 blink for detection
            if self.total >= thresh:
                self.total = 0
                return True
            return False
        except Exception as e:
            logger.error(f"Error in eye_blink: {str(e)}")
            return False

    def eye_aspect_ratio(self, eye):
        try:
            # Compute the euclidean distances between the two sets of
            # vertical eye landmarks (x, y)-coordinates
            A = math.dist(eye[1], eye[5])
            B = math.dist(eye[2], eye[4])
            
            # Compute the euclidean distance between the horizontal
            # eye landmark (x, y)-coordinates
            C = math.dist(eye[0], eye[3])
            
            # Compute the eye aspect ratio
            ear = (A + B) / (2.0 * C)
            
            # Return the eye aspect ratio
            return ear
        except Exception as e:
            logger.error(f"Error in eye_aspect_ratio: {str(e)}")
            return 0.3  # Return a default value that won't trigger blink detection

    def get_debug_info(self):
        """Return debug information about recent EAR values"""
        # Convert NumPy types to native Python types to ensure JSON serialization
        return {
            'recent_ear_values': [float(x) for x in self.recent_ears] if self.recent_ears else [],
            'baseline_ear': float(self.baseline_ear) if self.baseline_ear is not None else None,
            'min_ear_seen': float(self.min_ear_seen),
            'static_threshold': float(self.EYE_AR_THRESH),
            'counter': int(self.counter),
            'total_blinks': int(self.total),
            'blink_frames': [float(x) for x in self.blink_detected_frames] if self.blink_detected_frames else []
        }


class FaceOrientationDetector:
    """This class detects the orientation of a face in an image."""
    def __init__(self):
        # Adjusted thresholds for better accuracy
        self.left_threshold = 65  # Higher angle for left turn
        self.right_threshold = 30  # Increased from 20 to require more movement for right turn
        self.angle_diff_threshold = 15  # Minimum difference between angles
        
        # Store initial face position for comparison
        self.initial_landmarks = None
        self.initial_angles = None
        self.movement_threshold = 20  # Minimum movement required (in pixels)
        
    def calculate_angle(self, v1, v2):
        '''
        Calculate the angle between 2 vectors v1 and v2
        '''
        try:
            if isinstance(v1, torch.Tensor):
                v1 = v1.detach().cpu().numpy()
            else:
                v1 = np.array(v1)
                
            if isinstance(v2, torch.Tensor):
                v2 = v2.detach().cpu().numpy()
            else:
                v2 = np.array(v2)
                
            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            # Avoid division by zero
            if v1_norm == 0 or v2_norm == 0:
                return 0
                
            v1 = v1 / v1_norm
            v2 = v2 / v2_norm
            
            # Calculate dot product and clip to avoid numerical errors
            cosine = np.clip(np.dot(v1, v2), -1.0, 1.0)
            rad = np.arccos(cosine)
            degrees = np.degrees(rad)
            return float(np.round(degrees))  # Convert to native Python float
        except Exception as e:
            logger.error(f"Error in calculate_angle: {str(e)}")
            return 0
    
    def reset_initial_position(self):
        """Reset the initial position tracking"""
        self.initial_landmarks = None
        self.initial_angles = None
        
    def detect(self, landmarks, face_width=None, is_mirrored=True):
        '''
        Detects the orientation of a face based on landmarks.
        Parameters:
        landmarks (np.ndarray): A list of points representing the positions on the face
                              [left eye, right eye, nose, left mouth, right mouth].
        face_width (float): Optional width of the face for additional context
        is_mirrored (bool): Whether the video feed is mirrored (default: True)
        Returns:
        str: Returns the face orientation ('front', 'left', or 'right').
        '''
        try:
            # Extract key landmarks
            left_eye = np.array(landmarks[0])
            right_eye = np.array(landmarks[1])
            nose = np.array(landmarks[2])
            
            # Calculate vectors
            left2right_eye = right_eye - left_eye
            lefteye2nose = nose - left_eye
            left_angle = self.calculate_angle(left2right_eye, lefteye2nose)
            
            right2left_eye = left_eye - right_eye
            righteye2nose = nose - right_eye
            right_angle = self.calculate_angle(right2left_eye, righteye2nose)
            
            # Calculate the angle difference
            angle_diff = abs(left_angle - right_angle)
            
            # Calculate nose position relative to eye midpoint
            eye_midpoint = (left_eye + right_eye) / 2
            nose_offset = nose - eye_midpoint
            
            # Store initial position if not already set
            if self.initial_landmarks is None:
                self.initial_landmarks = [
                    np.array(landmarks[0]),
                    np.array(landmarks[1]),
                    np.array(landmarks[2])
                ]
                self.initial_angles = (left_angle, right_angle)
                logger.info(f"Initial face position set: left_angle={left_angle}, right_angle={right_angle}")
            
            # Calculate movement from initial position
            initial_left_eye = np.array(self.initial_landmarks[0])
            initial_right_eye = np.array(self.initial_landmarks[1])
            initial_nose = np.array(self.initial_landmarks[2])
            
            # Calculate movement distances
            left_eye_movement = float(np.linalg.norm(left_eye - initial_left_eye))
            right_eye_movement = float(np.linalg.norm(right_eye - initial_right_eye))
            nose_movement = float(np.linalg.norm(nose - initial_nose))
            
            # Calculate angle changes
            initial_left_angle, initial_right_angle = self.initial_angles
            left_angle_change = abs(left_angle - initial_left_angle)
            right_angle_change = abs(right_angle - initial_right_angle)
            
            # Determine orientation
            orientation = 'front'
            
            # For right orientation detection, require significant movement and angle change
            if left_angle < self.right_threshold and right_angle > self.left_threshold:
                # Basic angle-based detection
                orientation_by_angle = 'right'
                
                # But also verify with movement
                significant_movement = (left_eye_movement > self.movement_threshold or 
                                       right_eye_movement > self.movement_threshold or
                                       nose_movement > self.movement_threshold)
                
                significant_angle_change = (left_angle_change > 10 or right_angle_change > 10)
                
                if significant_movement and significant_angle_change:
                    orientation = 'right'
                    logger.info(f"Right turn detected: movement={nose_movement}, angle_changes=({left_angle_change}, {right_angle_change})")
                else:
                    logger.info(f"Right turn rejected: insufficient movement={nose_movement} or angle change=({left_angle_change}, {right_angle_change})")
            
            # For left orientation, we keep the existing logic as it's working well
            elif left_angle > self.left_threshold and right_angle < self.right_threshold:
                orientation = 'left'
                logger.info(f"Left turn detected: angles=({left_angle}, {right_angle})")
            elif angle_diff > self.angle_diff_threshold:
                # If there's a significant difference between angles
                if left_angle > right_angle:
                    orientation = 'left'
                else:
                    # For right, still verify movement
                    significant_movement = (left_eye_movement > self.movement_threshold or 
                                          right_eye_movement > self.movement_threshold or
                                          nose_movement > self.movement_threshold)
                    
                    if significant_movement:
                        orientation = 'right'
            elif nose_offset[0] < -5:  # Nose is to the left of eye midpoint
                orientation = 'left'
            elif nose_offset[0] > 5:   # Nose is to the right of eye midpoint
                # For right, still verify movement
                significant_movement = (left_eye_movement > self.movement_threshold or 
                                      right_eye_movement > self.movement_threshold or
                                      nose_movement > self.movement_threshold)
                
                if significant_movement:
                    orientation = 'right'
            
            # If mirrored, swap left and right
            if is_mirrored:
                if orientation == 'left':
                    orientation = 'right'
                elif orientation == 'right':
                    orientation = 'left'
            
            return orientation, float(left_angle), float(right_angle)  # Convert to native Python float
        except Exception as e:
            logger.error(f"Error in detect: {str(e)}")
            return 'front', 0.0, 0.0


class LivenessDetector:
    def __init__(self):
        # Initialize face detector
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Initialize blink detector
        self.blink_detector = BlinkDetector()
        
        # Initialize face orientation detector
        self.orientation_detector = FaceOrientationDetector()
        
        # Initialize face landmarks predictor
        self.predictor = dlib.shape_predictor(LANDMARK_PATH)
        
        # Track challenge changes to reset orientation detector
        self.current_challenge = None
    
    def detect_face(self, image):
        """Detect faces in the image"""
        try:
            faces = self.face_detector(image, 1)
            if len(faces) == 0:
                return None
            return faces[0]  # Return the first face detected
        except Exception as e:
            logger.error(f"Error in detect_face: {str(e)}")
            return None
    
    def get_face_box(self, face, expand_factor=1.2):
        """
        Convert dlib rectangle to [x, y, width, height] format
        Expand the box by the given factor for better face coverage
        """
        try:
            # Get original dimensions
            x = face.left()
            y = face.top()
            width = face.right() - face.left()
            height = face.bottom() - face.top()
            
            # Calculate expansion
            width_expansion = (expand_factor - 1) * width
            height_expansion = (expand_factor - 1) * height
            
            # Apply expansion
            new_x = max(0, int(x - width_expansion/2))
            new_y = max(0, int(y - height_expansion/2))
            new_width = int(width * expand_factor)
            new_height = int(height * expand_factor)
            
            # Convert to native Python types
            return [int(new_x), int(new_y), int(new_width), int(new_height)]
        except Exception as e:
            logger.error(f"Error in get_face_box: {str(e)}")
            return [0, 0, 100, 100]  # Return a default box
    
    def get_landmarks(self, image, face):
        """Get facial landmarks"""
        try:
            shape = self.predictor(image, face)
            shape = face_utils.shape_to_np(shape)
            
            # Extract key landmarks for orientation detection
            left_eye_center = np.mean(shape[36:42], axis=0).astype(int)
            right_eye_center = np.mean(shape[42:48], axis=0).astype(int)
            nose_tip = shape[30]
            
            return [left_eye_center, right_eye_center, nose_tip], shape
        except Exception as e:
            logger.error(f"Error in get_landmarks: {str(e)}")
            # Return default landmarks
            return [[0, 0], [10, 0], [5, 5]], np.zeros((68, 2), dtype=np.int32)
    
    def check_front_facing(self, image, face):
        """Check if face is looking directly at the camera"""
        try:
            landmarks, full_landmarks = self.get_landmarks(image, face)
            face_width = face.right() - face.left()
            orientation, left_angle, right_angle = self.orientation_detector.detect(landmarks, face_width)
            
            # For front-facing, we want the face to be centered and looking straight ahead
            # This means orientation should be 'front' and angles should be balanced
            is_front_facing = orientation == 'front'
            angle_balance = abs(left_angle - right_angle) < 20  # Angles should be similar
            
            # Also check if nose is centered between eyes
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            eye_midpoint = (left_eye + right_eye) / 2
            nose_offset = nose - eye_midpoint
            nose_centered = abs(nose_offset[0]) < 10  # Nose should be centered horizontally
            
            # Return detailed information for debugging and feedback
            success = is_front_facing or (angle_balance and nose_centered)
            return {
                'success': bool(success),  # Convert to native Python bool
                'detected': 'front' if success else 'not_front',
                'expected': 'front',
                'left_angle': float(left_angle),
                'right_angle': float(right_angle),
                'angle_diff': float(abs(left_angle - right_angle)),
                'nose_offset': float(nose_offset[0])
            }
        except Exception as e:
            logger.error(f"Error in check_front_facing: {str(e)}")
            return {
                'success': False,
                'detected': 'error',
                'expected': 'front',
                'left_angle': 0.0,
                'right_angle': 0.0,
                'angle_diff': 0.0,
                'nose_offset': 0.0
            }
    
    def check_orientation(self, image, face, expected_orientation):
        """Check if face orientation matches expected orientation"""
        try:
            landmarks, full_landmarks = self.get_landmarks(image, face)
            face_width = face.right() - face.left()
            orientation, left_angle, right_angle = self.orientation_detector.detect(landmarks, face_width)
            
            # Return detailed information for debugging and feedback
            return {
                'success': bool(orientation == expected_orientation),  # Convert to native Python bool
                'detected': orientation,
                'expected': expected_orientation,
                'left_angle': float(left_angle),
                'right_angle': float(right_angle)
            }
        except Exception as e:
            logger.error(f"Error in check_orientation: {str(e)}")
            return {
                'success': False,
                'detected': 'error',
                'expected': expected_orientation,
                'left_angle': 0.0,
                'right_angle': 0.0
            }
    
    def check_blink(self, image, face):
        """Check if blink is detected"""
        try:
            # Direct check first
            result = self.blink_detector.eye_blink(image, face, thresh=1)
            debug_info = self.blink_detector.get_debug_info()
            
            if result:
                # Update the shared result
                with blink_result_lock:
                    blink_result['detected'] = True
                    blink_result['timestamp'] = time.time()
                return {
                    'success': True,
                    'detected': 'blink',
                    'expected': 'blink',
                    'debug_info': debug_info
                }
            
            # Check if we have a recent blink detection result from background thread
            with blink_result_lock:
                current_time = time.time()
                if blink_result['detected'] and (current_time - blink_result['timestamp'] < 3.0):
                    # Reset the detection after using it
                    blink_result['detected'] = False
                    return {
                        'success': True,
                        'detected': 'blink',
                        'expected': 'blink',
                        'debug_info': debug_info
                    }
            
            # Add frame to queue for background processing if not successful yet
            if not blink_frames_queue.full():
                # Make a copy of the image to avoid memory issues
                blink_frames_queue.put((image.copy(), face))
            
            return {
                'success': False,
                'detected': 'no_blink',
                'expected': 'blink',
                'debug_info': debug_info
            }
        except Exception as e:
            logger.error(f"Error in check_blink: {str(e)}")
            return {
                'success': False,
                'detected': 'error',
                'expected': 'blink',
                'debug_info': {}
            }
    
    def process_frame(self, image, challenge):
        """Process a single frame for a specific challenge"""
        try:
            # Check if challenge has changed
            if challenge != self.current_challenge:
                # Reset orientation detector when challenge changes
                self.orientation_detector.reset_initial_position()
                self.current_challenge = challenge
                logger.info(f"Challenge changed to: {challenge}")
            
            # Detect face
            face = self.detect_face(image)
            if face is None:
                return {
                    'success': False,
                    'message': 'No face detected. Please ensure your face is clearly visible.',
                    'face_box': None
                }
            
            # Get face box for visualization - always return this with expanded size
            face_box = self.get_face_box(face, expand_factor=1.2)
            
            # Process based on challenge type
            if challenge == 'front':
                # New challenge: Look directly very close to the camera
                result = self.check_front_facing(image, face)
                if result['success']:
                    message = 'Face centered and looking at camera!'
                else:
                    message = 'Please look directly very close to the camera'
                    
            elif challenge == 'right':
                result = self.check_orientation(image, face, 'right')
                if result['success']:
                    message = 'Face turned right detected!'
                else:
                    if result['detected'] == 'front':
                        message = 'Please turn your face more to the right'
                    elif result['detected'] == 'left':
                        message = 'Wrong direction! Please turn your face to the right'
                    else:
                        message = f'Please turn your face right'
                        
            elif challenge == 'left':
                result = self.check_orientation(image, face, 'left')
                if result['success']:
                    message = 'Face turned left detected!'
                else:
                    if result['detected'] == 'front':
                        message = 'Please turn your face more to the left'
                    elif result['detected'] == 'right':
                        message = 'Wrong direction! Please turn your face to the left'
                    else:
                        message = f'Please turn your face left'
                        
            elif challenge == 'blink':
                result = self.check_blink(image, face)
                if result['success']:
                    message = 'Blink detected!'
                else:
                    message = 'Please blink your eyes'
            else:
                result = {'success': False}
                message = f'Unknown challenge: {challenge}'
            
            # Add debug info for orientation challenges
            debug_info = {}
            if challenge in ['right', 'left', 'front']:
                debug_info = {
                    'left_angle': float(result.get('left_angle', 0)),
                    'right_angle': float(result.get('right_angle', 0)),
                    'detected': result.get('detected', '')
                }
                if challenge == 'front':
                    debug_info['angle_diff'] = float(result.get('angle_diff', 0))
                    debug_info['nose_offset'] = float(result.get('nose_offset', 0))
            elif challenge == 'blink':
                debug_info = result.get('debug_info', {})
            
            # Force garbage collection to prevent memory issues
            gc.collect()
            
            # Ensure all values are JSON serializable
            return {
                'success': bool(result['success']),  # Convert to native Python bool
                'message': message,
                'face_box': face_box,  # Include face box but no landmarks
                'debug_info': debug_info
            }
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return {
                'success': False,
                'message': f'Error processing frame: {str(e)}',
                'face_box': None,
                'debug_info': {}
            }


# Background thread for blink detection
def blink_detection_worker():
    # Create a dedicated blink detector for this thread
    detector = BlinkDetector()
    
    while True:
        try:
            # Get a frame from the queue
            image, face = blink_frames_queue.get(timeout=1)
            
            # Process the frame for blink detection
            result = detector.eye_blink(image, face, thresh=1)
            
            # If blink detected, update the shared result
            if result:
                with blink_result_lock:
                    blink_result['detected'] = True
                    blink_result['timestamp'] = time.time()
                    logger.info("Blink detected in background thread")
            
            # Mark task as done
            blink_frames_queue.task_done()
            
            # Clean up to prevent memory leaks
            del image
            gc.collect()
            
        except queue.Empty:
            # Queue is empty, just continue
            pass
        except Exception as e:
            logger.error(f"Error in blink detection worker: {str(e)}")
            # Continue processing even if there's an error


# Initialize the liveness detector
liveness_detector = None

# Start the background blink detection thread
blink_thread = threading.Thread(target=blink_detection_worker, daemon=True)
blink_thread.start()

# Create a background thread for model loading
def load_model_in_background():
    global liveness_detector
    if not os.path.exists(LANDMARK_PATH):
        logger.warning("Model file not found. Please download it first.")
        return
    
    logger.info("Loading model in background...")
    liveness_detector = LivenessDetector()
    logger.info("Model loaded successfully!")

# Start model loading in background when app starts
model_loading_thread = threading.Thread(target=load_model_in_background)
model_loading_thread.daemon = True
model_loading_thread.start()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/api/process-frame', methods=['POST', 'OPTIONS'])
def process_frame():
    """Process a single frame for real-time feedback"""
    global liveness_detector
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Check if landmark model exists, if not, return error
        if not os.path.exists(LANDMARK_PATH):
            return jsonify({
                'success': False,
                'message': 'Landmark model not found. Please download the model first.',
                'face_box': None
            }), 400
        
        # Wait for model to load if it's not ready yet
        start_time = time.time()
        while liveness_detector is None:
            time.sleep(0.1)
            if time.time() - start_time > 5:  # Timeout after 5 seconds
                return jsonify({
                    'success': False,
                    'message': 'Model is still loading. Please try again in a moment.',
                    'face_box': None
                }), 503
        
        # Get data from request
        data = request.json
        challenge = data.get('challenge')
        frame_base64 = data.get('frame')
        
        # Extract the base64 data
        image_data = re.sub('^data:image/.+;base64,', '', frame_base64)
        image_bytes = base64.b64decode(image_data)
        
        # Convert to image
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # Convert to RGB if needed
        if image_np.shape[2] == 4:  # RGBA
            image_np = image_np[:, :, :3]
        
        # Process the frame
        result = liveness_detector.process_frame(image_np, challenge)
        
        # Log blink detection results for debugging
        if challenge == 'blink':
            debug_info = result.get('debug_info', {})
            logger.info(f"Blink detection: success={result['success']}, " +
                      f"baseline={debug_info.get('baseline_ear')}, " +
                      f"min_ear={debug_info.get('min_ear_seen')}")
        
        # Clean up to prevent memory leaks
        del image
        del image_np
        gc.collect()
        
        # Use custom JSON encoder to handle NumPy types
        return custom_jsonify(result)
            
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return jsonify({
            'success': False, 
            'message': f'Server error: {str(e)}',
            'face_box': None
        }), 500

@app.route('/api/liveness-detection', methods=['POST', 'OPTIONS'])
def liveness_detection():
    """Process liveness detection frames for final verification"""
    global liveness_detector
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Check if landmark model exists, if not, return error
        if not os.path.exists(LANDMARK_PATH):
            return jsonify({
                'success': False,
                'message': 'Landmark model not found. Please download the model first.'
            }), 400
        
        # Wait for model to load if it's not ready yet
        start_time = time.time()
        while liveness_detector is None:
            time.sleep(0.1)
            if time.time() - start_time > 5:  # Timeout after 5 seconds
                return jsonify({
                    'success': False,
                    'message': 'Model is still loading. Please try again in a moment.'
                }), 503
        
        # Get data from request
        data = request.json
        frames = data.get('frames', [])
        
        if not frames:
            return jsonify({'success': False, 'message': 'No frames provided'}), 400
        
        # Process each frame
        results = []
        for frame_data in frames:
            challenge = frame_data.get('challenge')
            frame_base64 = frame_data.get('frame')
            
            # Extract the base64 data
            image_data = re.sub('^data:image/.+;base64,', '', frame_base64)
            image_bytes = base64.b64decode(image_data)
            
            # Convert to image
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            # Convert to RGB if needed
            if image_np.shape[2] == 4:  # RGBA
                image_np = image_np[:, :, :3]
            
            # Process the frame
            result = liveness_detector.process_frame(image_np, challenge)
            results.append({
                'challenge': challenge,
                'success': bool(result['success']),  # Convert to native Python bool
                'message': result['message']
            })
            
            # Clean up to prevent memory leaks
            del image
            del image_np
            gc.collect()
        
        # Check if all challenges were successful
        all_successful = all(result['success'] for result in results)
        
        if all_successful:
            return custom_jsonify({
                'success': True,
                'message': 'Liveness verification successful',
                'results': results
            })
        else:
            # Find the first failed challenge
            failed = next((r for r in results if not r['success']), None)
            return custom_jsonify({
                'success': False,
                'message': f'Liveness verification failed: {failed["message"]}',
                'results': results
            })
            
    except Exception as e:
        logger.error(f"Error in liveness detection: {str(e)}")
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

@app.route('/success')
def success():
    """Success page after successful verification"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Verification Successful</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                text-align: center;
                padding: 50px 20px;
                background-color: #f5f5f5;
            }
            .success-container {
                max-width: 500px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #4CAF50;
            }
            .btn {
                display: inline-block;
                background-color: #4CAF50;
                color: white;
                padding: 12px 30px;
                text-decoration: none;
                border-radius: 4px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="success-container">
            <h1>Verification Successful!</h1>
            <p>Your liveness verification has been completed successfully.</p>
            <a href="/" class="btn">Start Over</a>
        </div>
    </body>
    </html>
    """

@app.route('/download-model')
def download_model():
    """Download the required landmark model"""
    try:
        import urllib.request
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(LANDMARK_PATH), exist_ok=True)
        
        # URL for the shape predictor model
        model_url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks_GTX.dat.bz2"
        
        # Download the compressed model
        compressed_model_path = 'model/shape_predictor_68_face_landmarks_GTX.dat.bz2'
        urllib.request.urlretrieve(model_url, compressed_model_path)
        
        # Decompress the model
        import bz2
        with bz2.BZ2File(compressed_model_path) as f_in, open(LANDMARK_PATH, 'wb') as f_out:
            f_out.write(f_in.read())
        
        # Remove the compressed file
        os.remove(compressed_model_path)
        
        # Initialize the detector in background
        if not model_loading_thread.is_alive():
            new_thread = threading.Thread(target=load_model_in_background)
            new_thread.daemon = True
            new_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Model downloaded successfully'
        })
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error downloading model: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Use 0.0.0.0 to make the server accessible externally
    # Changed port to 5555 as requested
    
    # Check if SSL certificates exist
    cert_file = 'cert/cert.pem'
    key_file = 'cert/key.pem'
    
    # Create cert directory if it doesn't exist
    os.makedirs(os.path.dirname(cert_file), exist_ok=True)
    
    # Generate self-signed certificates if they don't exist
    if not (os.path.exists(cert_file) and os.path.exists(key_file)):
        logger.info("Generating self-signed SSL certificates...")
        from OpenSSL import crypto
        
        # Create a key pair
        k = crypto.PKey()
        k.generate_key(crypto.TYPE_RSA, 2048)
        
        # Create a self-signed cert
        cert = crypto.X509()
        cert.get_subject().C = "US"
        cert.get_subject().ST = "State"
        cert.get_subject().L = "City"
        cert.get_subject().O = "Organization"
        cert.get_subject().OU = "Organizational Unit"
        cert.get_subject().CN = "localhost"
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(10*365*24*60*60)  # 10 years
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(k)
        cert.sign(k, 'sha256')
        
        # Save certificate and key
        with open(cert_file, "wb") as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        with open(key_file, "wb") as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))
        
        logger.info(f"Self-signed certificates generated: {cert_file}, {key_file}")
    
    # Create SSL context
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_file, key_file)
    
    # Run with HTTPS
    logger.info(f"Starting server with HTTPS on port 5555")
    app.run(host='0.0.0.0', port=5555, ssl_context=context, debug=True)
