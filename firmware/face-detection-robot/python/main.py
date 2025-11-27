from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI
import cv2
import numpy as np
import onnxruntime as ort
import base64
import threading
import time
from dataclasses import dataclass
from typing import List, Callable, Optional

# Settings
MODEL_PATH = "model_with_nms.onnx"
CAMERA_INDEX = 0
TARGET_W = 416
TARGET_H = 416
CONFIDENCE_THRESHOLD = 0.8
WEB_UI_ENABLED = False

# Class names
CLASS_NAMES = [
    "angry", 
    "disgust", 
    "fear", 
    "happy", 
    "neutral", 
    "sad", 
    "surprise",
]

################################################################################
# Data Structure for Inference Results

@dataclass
class DetectionResult:
    """Container for detection results"""
    frame: np.ndarray          # Preprocessed frame (BGR)
    boxes: List[List[float]]   # List of [x1, y1, x2, y2]
    scores: List[float]        # Confidence scores
    classes: List[int]         # Class IDs
    class_names: List[str]     # Class names (for convenience)
    inference_time_ms: float   # How long inference took
    timestamp: float           # When this was captured
    frame_number: int          # Sequential frame counter

################################################################################
# Preprocessing Functions

def preprocess_image(
    img,
    target_w=640,
    target_h=640,
    bg_color=(0, 0, 0),
):
    """Load and preprocess an image for training"""
    # Convert image to RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Calculate new width and height in order to maintain aspect ratio
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    # Resize with aspect ratio maintained
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute padding (center the image)
    pad_w, pad_h = target_w - new_w, target_h - new_h
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2

    # Apply border padding (black background)
    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=bg_color,
    )

    return img_padded, scale, pad_left, pad_top

def img_to_np(img):
    """Convert an image to a normalized NumPy array"""
    # Ensure 3-channel RGB shape
    if img.ndim == 2:
        # Grayscale image (H, W), copy image to all three channels
        img = np.repeat(img[..., None], 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 1:
        # Single-channel image (H, W, 1), copy image to all three channels
        img = np.repeat(img, 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 3:
        # Already RGB, do nothing
        pass
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    # Reorder image (H, W, C) to (C, H, W) format
    img = np.transpose(img, (2, 0, 1))

    # Convert to float32 and normalize from [0, 255] to [0.0, 1.0]
    img = img.astype(np.float32) / 255.0

    return img

def draw_bounding_boxes(img, boxes, scores, classes, class_names):
    """Draw bounding boxes on the image"""
    img_with_boxes = img.copy()
    
    # Go through all bounding boxes
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(coord) for coord in boxes[i]]
        confidence = scores[i]
        class_id = classes[i]
        class_name = class_names[class_id]
        
        # Color code by emotion
        color_map = {
            "happy": (0, 255, 0),       # Green
            "neutral": (255, 255, 0),   # Yellow
            "sad": (255, 0, 0),         # Blue
            "angry": (0, 0, 255),       # Red
            "surprise": (255, 0, 255),  # Magenta
            "fear": (0, 165, 255),      # Orange
            "disgust": (128, 0, 128),   # Purple
        }
        color = color_map.get(class_name, (0, 255, 0))
        
        # Draw rectangle
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        label = f"{class_name}: {confidence:.2f}"
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            img_with_boxes,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            img_with_boxes,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return img_with_boxes

def get_largest_detection(result: DetectionResult):
    """Find the index of the largest bounding box by area."""
    if len(result.boxes) == 0:
        return None

    # Find box with largest area
    largest_idx = 0
    largest_area = 0
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        
        if area > largest_area:
            largest_area = area
            largest_idx = i
    
    return largest_idx

################################################################################
# Class to handle inference on webcam images

class WebcamInference:
    def __init__(
        self,
        model_path,
        camera_index=0, 
        target_width=640, 
        target_height=640,
        confidence_threshold=0.1,
    ):
        """
        Initialize the webcam inference engine with callback support.
        """
        self.camera_index = camera_index
        self.target_width = target_width
        self.target_height = target_height
        self.confidence_threshold = confidence_threshold
        self.result = None
        
        # Thread control
        self.running = False
        self.capture = None
        self.thread = None
        
        # Statistics
        self.frame_count = 0
        self.fps = 0
        
        # Use CUDA ONNX Runtime if available, fall back to CPU
        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"Available providers: {ort.get_available_providers()}")
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("Using GPU acceleration")
        else:
            print("Using CPU only")

        # Load model into ONNX Runtime
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model input and output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape
        
        # Print model info
        print(f"Model loaded: {model_path}")
        print(f"Input: {self.input_name}, shape: {self.input_shape}")
        print(f"Output: {self.output_name}, shape: {self.output_shape}")
    
    def start(self):
        """Start the webcam capture and inference thread."""
        # Initialize camera
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            print(f"Error: Could not open webcam at index {self.camera_index}")
            return False
            
        # Get actual 
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera native resolution: {actual_width}x{actual_height}")
        print(f"Preprocess to: {self.target_width}x{self.target_height}")
        
        # Start inference thread
        self.running = True
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

        return True
    
    def stop(self):
        """Stop the inference thread and release the webcam."""
        self.running = False
        
        # Wait for thread to stop
        if self.thread:
            self.thread.join(timeout=2.0)
        
        # Stop webcam
        if self.capture:
            self.capture.release()
        
        print("Inference engine stopped")

    def _inference_loop(self):
        """
        Main inference loop: capture image, preprocess, inference, actions
        """
        print("Inference started")
        
        # Run thread
        fps_frame_count = 0
        last_fps_calc_time = time.time()
        while self.running:
            loop_start = time.time()
            self.result = None

            # Calculate FPS every second
            fps_frame_count += 1
            fps_timestamp = time.time()
            elapsed = fps_timestamp - last_fps_calc_time
            if elapsed >= 1.0:
                self.fps = fps_frame_count / elapsed
                fps_frame_count = 0
                last_fps_calc_time = fps_timestamp
            
            # Capture frame
            ret, frame = self.capture.read()
            if not ret:
                print(f"Error: Failed to capture image from webcam")
                time.sleep(0.5)
                continue
            
            self.frame_count += 1
            
            # Letterbox and scale image
            frame_preprocessed, _, _, _ = preprocess_image(
                frame,
                target_w=self.target_width,
                target_h=self.target_height,
                bg_color=(0, 0, 0)
            )
            
            # Convert to model input
            img_np = img_to_np(frame_preprocessed)
            img_np_batched = np.expand_dims(img_np, axis=0)
            
            # Run inference
            inference_start = time.time()
            outputs = self.session.run(
                [self.output_name], 
                {self.input_name: img_np_batched}
            )
            inference_time = (time.time() - inference_start) * 1000
            
            # Parse detections
            detections = outputs[0][0]
            pred_boxes = []
            pred_scores = []
            pred_classes = []
            for detection in detections:
                x1, y1, x2, y2, confidence, class_id = detection

                # Append bounding box info if above confidence threshold
                if confidence > self.confidence_threshold:
                    pred_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                    pred_scores.append(float(confidence))
                    pred_classes.append(int(class_id))
            
            # Build class names list
            pred_class_names = [CLASS_NAMES[c] for c in pred_classes]
            
            # Create result object
            self.result = DetectionResult(
                frame=frame_preprocessed,
                boxes=pred_boxes,
                scores=pred_scores,
                classes=pred_classes,
                class_names=pred_class_names,
                inference_time_ms=inference_time,
                timestamp=time.time(),
                frame_number=self.frame_count
            )

            # Actions: animate the robot head and optionally update web UI 
            self._animate_head()
            if WEB_UI_ENABLED:
                self._update_web_ui()

    def _animate_head(self):
        """Move servos and update LED ring to respond to face emotion"""
        # Get largest bounding box
        largest_idx = get_largest_detection(self.result)
        if largest_idx is None:
            return
        
        # Calculate normalized center
        box = self.result.boxes[largest_idx]
        x_center_norm = ((box[0] + box[2]) / 2) / self.result.frame.shape[1]
        y_center_norm = ((box[1] + box[3]) / 2) / self.result.frame.shape[0]
        
        # Don't move if in dead zone
        if abs(x_center_norm - 0.5) < 0.05 and abs(y_center_norm - 0.5) < 0.05:
            return
        
        # Send info to MCU
        Bridge.call(
            "animate_head", 
            x_center_norm,
            y_center_norm, 
            self.result.classes[largest_idx],
            self.result.scores[largest_idx],
        );


    def _update_web_ui(self):
        """Update the web UI with image annotated with bounding boxes"""
        # Draw bounding boxes
        frame_with_boxes = draw_bounding_boxes(
            self.result.frame,
            self.result.boxes,
            self.result.scores,
            self.result.classes,
            CLASS_NAMES
        )
        
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            print("Web UI: failed to encode frame")
            return
        
        # Convert to base64
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        # Send to web UI
        ui.send_message('webcam_frame', {
            'image': jpg_as_text,
            'timestamp': self.result.timestamp,
            'frame_number': self.result.frame_number,
            'detections': len(self.result.boxes),
            'inference_time_ms': round(self.result.inference_time_ms, 1),
            'fps': round(self.fps, 1),
            'detections_detail': [
                {
                    'class': self.result.class_names[i],
                    'confidence': round(self.result.scores[i], 2),
                    'box': [round(x) for x in self.result.boxes[i]]
                }
                for i in range(len(self.result.boxes))
            ]
        })

################################################################################
# Main

# Initialize the Web UI
ui = WebUI()

# Create inference engine
inference = WebcamInference(
    model_path=MODEL_PATH,
    camera_index=CAMERA_INDEX,
    target_width=TARGET_W,
    target_height=TARGET_H,
    confidence_threshold=CONFIDENCE_THRESHOLD,
)

# Start inference
if not inference.start():
    print("Error: Failed to start inference thread")

# Run the Arduino App Framework and all bricks we loaded (blocks forever)
App.run()