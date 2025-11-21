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

# Model settings
MODEL_PATH = "model_with_nms.onnx"
TARGET_W = 416
TARGET_H = 416
CONFIDENCE_THRESHOLD = 0.01

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
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w, pad_h = target_w - new_w, target_h - new_h
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2

    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=bg_color,
    )

    return img_padded, scale, pad_left, pad_top

def img_to_np(img):
    """Convert an image to a normalized NumPy array"""
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32) / 255.0

    return img

def draw_bounding_boxes(img, boxes, scores, classes, class_names):
    """Draw bounding boxes on the image."""
    img_with_boxes = img.copy()
    
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(coord) for coord in boxes[i]]
        confidence = scores[i]
        class_id = classes[i]
        class_name = class_names[class_id]
        
        # Color-code by emotion
        color_map = {
            "happy": (0, 255, 0),      # Green
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
            (255, 255, 255),  # White text
            1
        )
    
    return img_with_boxes

################################################################################
# Callback-based Webcam Streamer

class WebcamInference:
    def __init__(
        self, 
        camera_index=1, 
        target_width=416, 
        target_height=416,
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD
    ):
        """
        Initialize the webcam inference engine with callback support.
        
        Args:
            camera_index: Camera device index (default 1)
            target_width: Target width for preprocessing (default 416)
            target_height: Target height for preprocessing (default 416)
            model_path: Path to ONNX model
            confidence_threshold: Minimum confidence for detections
        """
        self.camera_index = camera_index
        self.target_width = target_width
        self.target_height = target_height
        self.confidence_threshold = confidence_threshold
        
        # Thread control
        self.running = False
        self.capture = None
        self.thread = None
        
        # Callback registry
        self.callbacks: List[Callable[[DetectionResult], None]] = []
        
        # Statistics
        self.frame_count = 0
        
        # Load ONNX model
        print("Loading ONNX model...")
        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"Available providers: {ort.get_available_providers()}")
        
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("Using GPU acceleration")
        else:
            print("Using CPU only")
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape
        
        print(f"Model loaded: {model_path}")
        print(f"Input: {self.input_name}, shape: {self.input_shape}")
        print(f"Output: {self.output_name}, shape: {self.output_shape}")
    
    def on_detection(self, callback: Callable[[DetectionResult], None]):
        """
        Register a callback to be called when inference completes.
        
        Args:
            callback: Function that takes a DetectionResult and returns None
        
        Example:
            def my_callback(result: DetectionResult):
                print(f"Found {len(result.boxes)} objects")
                # Update servos based on result.boxes
            
            inference.on_detection(my_callback)
        """
        self.callbacks.append(callback)
        print(f"Registered callback: {callback.__name__}")
    
    def start(self):
        """Start the webcam capture and inference thread."""
        # Initialize camera
        self.capture = cv2.VideoCapture(self.camera_index)
        
        if not self.capture.isOpened():
            print(f"Warning: Could not open camera at index {self.camera_index}")
            for idx in [0, 1, 2]:
                if idx == self.camera_index:
                    continue
                print(f"Trying camera index {idx}...")
                self.capture = cv2.VideoCapture(idx)
                if self.capture.isOpened():
                    print(f"Success! Using camera index {idx}")
                    self.camera_index = idx
                    break
            
        if not self.capture.isOpened():
            print("Error: Could not open any webcam")
            return False
            
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera native resolution: {actual_width}x{actual_height}")
        print(f"Will preprocess to: {self.target_width}x{self.target_height}")
        
        # Start inference thread
        self.running = True
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
        
        print(f"Inference engine started with {len(self.callbacks)} callback(s)")
        return True
    
    def _inference_loop(self):
        """
        Main inference loop: capture → preprocess → inference → callbacks
        Runs as fast as possible.
        """
        consecutive_failures = 0
        max_failures = 30
        
        print("[Inference] Started")
        
        while self.running and consecutive_failures < max_failures:
            loop_start = time.time()
            
            # Step 1: Capture frame
            ret, frame = self.capture.read()
            
            if not ret:
                consecutive_failures += 1
                print(f"[Inference] Failed to capture ({consecutive_failures}/{max_failures})")
                time.sleep(0.5)
                continue
            
            consecutive_failures = 0
            self.frame_count += 1
            
            # Step 2: Preprocess
            frame_preprocessed, scale, pad_left, pad_top = preprocess_image(
                frame,
                target_w=self.target_width,
                target_h=self.target_height,
                bg_color=(0, 0, 0)
            )
            
            # Convert to model input
            img_np = img_to_np(frame_preprocessed)
            img_np_batched = np.expand_dims(img_np, axis=0)
            
            # Step 3: Inference
            inference_start = time.time()
            outputs = self.session.run(
                [self.output_name], 
                {self.input_name: img_np_batched}
            )
            inference_time = (time.time() - inference_start) * 1000
            
            # Step 4: Parse detections
            detections = outputs[0][0]
            
            pred_boxes = []
            pred_scores = []
            pred_classes = []
            
            for detection in detections:
                x1, y1, x2, y2, confidence, class_id = detection
                
                if confidence > self.confidence_threshold:
                    pred_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                    pred_scores.append(float(confidence))
                    pred_classes.append(int(class_id))
            
            # Build class names list
            pred_class_names = [CLASS_NAMES[c] for c in pred_classes]
            
            # Create result object
            result = DetectionResult(
                frame=frame_preprocessed,
                boxes=pred_boxes,
                scores=pred_scores,
                classes=pred_classes,
                class_names=pred_class_names,
                inference_time_ms=inference_time,
                timestamp=time.time(),
                frame_number=self.frame_count
            )
            
            # Step 5: Call all registered callbacks
            for callback in self.callbacks:
                try:
                    callback(result)
                except Exception as e:
                    print(f"[Inference] Error in callback {callback.__name__}: {e}")
            
            # Log stats periodically
            if self.frame_count % 30 == 0:
                total_time = (time.time() - loop_start) * 1000
                print(f"\n[Inference] Frame {self.frame_count}")
                print(f"  Detections: {len(pred_boxes)}")
                if len(pred_boxes) > 0:
                    for i in range(len(pred_boxes)):
                        print(f"    {pred_class_names[i]}: {pred_scores[i]:.2f}")
                print(f"  Inference: {inference_time:.1f}ms")
                print(f"  Total loop: {total_time:.1f}ms")
                print(f"  FPS: {1000/total_time:.1f}")
        
        if consecutive_failures >= max_failures:
            print("[Inference] Too many failures, stopping")
        
        print("[Inference] Stopped")
    
    def stop(self):
        """Stop the inference thread and release the webcam."""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if self.capture:
            self.capture.release()
        
        print("Inference engine stopped")

################################################################################
# Callback Functions

def update_web_ui(result: DetectionResult):
    """
    Callback to update the web UI with detection results.
    This runs every time inference completes.
    """
    # Draw bounding boxes
    frame_with_boxes = draw_bounding_boxes(
        result.frame,
        result.boxes,
        result.scores,
        result.classes,
        CLASS_NAMES
    )
    
    # Encode as JPEG
    ret, buffer = cv2.imencode('.jpg', frame_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    if not ret:
        print("[Web UI] Failed to encode frame")
        return
    
    # Convert to base64
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    
    # Send to web UI
    ui.send_message('webcam_frame', {
        'image': jpg_as_text,
        'timestamp': result.timestamp,
        'frame_number': result.frame_number,
        'detections': len(result.boxes),
        'inference_time_ms': round(result.inference_time_ms, 1),
        'detections_detail': [
            {
                'class': result.class_names[i],
                'confidence': round(result.scores[i], 2),
                'box': [round(x) for x in result.boxes[i]]
            }
            for i in range(len(result.boxes))
        ]
    })

def control_servos(result: DetectionResult):
    """
    Callback to control servos based on detection results.
    This is where you'll add Arduino bridge logic later.
    """
    # TODO: Add servo control logic here
    # For now, just log what we would do
    
    if len(result.boxes) == 0:
        # No detections - could center servos or hold position
        pass
    else:
        # Get the first (highest confidence) detection
        box = result.boxes[0]
        class_name = result.class_names[0]
        confidence = result.scores[0]
        
        # Calculate center of bounding box
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        
        # Normalize to [0, 1] range (for servo angles later)
        x_normalized = x_center / result.frame.shape[1]  # width
        y_normalized = y_center / result.frame.shape[0]  # height
        
        # Log what we would do (replace with actual servo control later)
        if result.frame_number % 30 == 0:  # Don't spam logs
            print(f"\n[Servo Control] Would track: {class_name}")
            print(f"  Box center: ({x_center:.0f}, {y_center:.0f})")
            print(f"  Normalized: ({x_normalized:.2f}, {y_normalized:.2f})")
            # Future: arduino_bridge.set_servo_position(x_normalized, y_normalized)

################################################################################
# Main

# Initialize the Web UI
ui = WebUI()

# Create inference engine
inference = WebcamInference(
    camera_index=1,
    target_width=TARGET_W,
    target_height=TARGET_H,
    model_path=MODEL_PATH,
    confidence_threshold=CONFIDENCE_THRESHOLD
)

# Register callbacks
inference.on_detection(update_web_ui)
inference.on_detection(control_servos)

# Start inference
if inference.start():
    print("\nInference engine initialized successfully")
    print("Registered callbacks:")
    print("  1. update_web_ui - sends frames to browser")
    print("  2. control_servos - tracks detections (stub for now)")
else:
    print("Failed to initialize inference engine")

# Run the Arduino App Framework and all bricks we loaded (blocks forever)
App.run()