import os
import time

import cv2
import numpy as np
import onnxruntime as ort

# Settings
IMG_PATH = "test.jpg"
MODEL_PATH = "model_with_nms.onnx"
TARGET_W = 416
TARGET_H = 416
CONFIDENCE_THRESHOLD = 0.01
OUTPUT_PATH = "out.jpg"

# Class names (get from original labels in data)
class_names = [
    "angry", 
    "disgust", 
    "fear", 
    "happy", 
    "neutral", 
    "sad", 
    "surprise",
]

################################################################################
# Functions

def preprocess_image(
    img,
    target_w,
    target_h,
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
    img_resized = cv2.resize(
        img, 
        (new_w, new_h), 
        interpolation=cv2.INTER_LINEAR
    )

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

################################################################################
# Main

def main():
    """Main entrypoint"""

    # Get ONNX Runtime info
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Build info: {ort.get_build_info()}")
    print(f"Available providers: {ort.get_available_providers()}")
    
    # Check if CUDA is available, default to CPU if not
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("Using GPU acceleration")
    else:
        print("Using CPU only")

    session = ort.InferenceSession(
        "model.onnx",
        providers=providers
    )

    # Load ONNX model
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )

    # Get model info
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape

    # Print model info
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Input: {input_name}, shape: {input_shape}")
    print(f"Output: {output_name}, shape: {output_shape}")

    # Load the image
    img = cv2.imread(str(IMG_PATH), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: Failed to load image from {IMG_PATH}")
        exit(1)
    print(f"Original image shape: {img.shape}")
    
    # Start timing
    timestamp_start = time.perf_counter()

    # Preprocess the image
    img_proc, scale, pad_left, pad_top = preprocess_image(
        img, 
        TARGET_W, 
        TARGET_H,
    )
    print(f"Processed image shape: {img_proc.shape}")
    print(f"Scale factor: {scale:.4f}")
    print(f"Padding: left={pad_left}, top={pad_top}")

    # Convert to NumPy array with correct shape (batch size of 1)
    img_np = img_to_np(img_proc)
    img_np_batched = np.expand_dims(img_np, axis=0)
    print(f"Batched image shape: {img_np_batched.shape}")

    # Run inference
    timestamp_inference = time.perf_counter()
    outputs = session.run([output_name], {input_name: img_np_batched})
    inference_time = (time.perf_counter() - timestamp_inference) * 1000

    # Get just the detections
    detections = outputs[0][0]

    # Format: [x1, y1, x2, y2, confidence, class_id]
    pred_boxes = []
    pred_scores = []
    pred_classes = []

    # Get each of the bounding boxes
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        
        # Filter out anything below our confidence threshold
        if confidence > CONFIDENCE_THRESHOLD:
            pred_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
            pred_scores.append(float(confidence))
            pred_classes.append(int(class_id))

    # Print the raw detections (note: NMS is included in ONNX model)
    for i in range(len(pred_boxes)):
        print(f"{pred_classes[i]} ({class_names[pred_classes[i]]}) with confidence " \
            f"{round(pred_scores[i], 2)} at {[round(v) for v in pred_boxes[i]]}")
        
    # Log total time
    total_time = (time.perf_counter() - timestamp_start) * 1000
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Total processing time: {total_time:.2f} ms")
    print(f"Pre/post-processing overhead: {total_time - inference_time:.2f} ms")

if __name__ == "__main__":
    main()