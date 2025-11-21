# SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
#
# SPDX-License-Identifier: MPL-2.0

from arduino.app_utils import *
from arduino.app_bricks.web_ui import WebUI
import cv2
import base64
import threading
import time

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

class WebcamStreamer:
    def __init__(self, fps=10, camera_index=1, target_width=640, target_height=640):
        """
        Initialize the webcam streamer.
        
        Args:
            fps: Frames per second to stream (default 10)
            camera_index: Camera device index (default 1, matching Arduino's /dev/video1)
            target_width: Target width for preprocessing (default 640)
            target_height: Target height for preprocessing (default 640)
        """
        self.fps = fps
        self.camera_index = camera_index
        self.target_width = target_width
        self.target_height = target_height
        self.frame_interval = 1.0 / fps
        self.running = False
        self.capture = None
        self.thread = None
        
    def start(self):
        """Start the webcam capture and streaming thread."""
        # Try the specified camera index first
        self.capture = cv2.VideoCapture(self.camera_index)
        
        if not self.capture.isOpened():
            print(f"Warning: Could not open camera at index {self.camera_index}")
            # Try alternative indices
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
            
        # Set camera resolution
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Get actual resolution (camera might not support requested resolution)
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_width}x{actual_height}")
        
        # Start the streaming thread
        self.running = True
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        
        print(f"Webcam streaming started at {self.fps} fps from /dev/video{self.camera_index}")
        return True
        
    def _stream_loop(self):
        """
        Main loop that runs in a separate thread.
        Captures frames and sends them to the web UI.
        """
        consecutive_failures = 0
        max_failures = 30  # Stop after 30 consecutive failures
        
        while self.running and consecutive_failures < max_failures:
            start_time = time.time()
            
            # Capture a frame from the webcam
            ret, frame = self.capture.read()
            
            if not ret:
                consecutive_failures += 1
                print(f"Warning: Failed to capture frame ({consecutive_failures}/{max_failures})")
                time.sleep(0.5)  # Wait a bit before retrying
                continue
            
            # Reset failure counter on success
            consecutive_failures = 0

            # Preprocess the frame with letterboxing
            frame_preprocessed, scale, pad_left, pad_top = preprocess_image(
                frame,
                target_w=self.target_width,
                target_h=self.target_height,
                bg_color=(0, 0, 0)  # Black letterboxing
            )
            
            # Encode the frame directly as JPEG (OpenCV handles BGR->RGB during encoding)
            ret, buffer = cv2.imencode(
                '.jpg', 
                frame_preprocessed, 
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if not ret:
                print("Error: Failed to encode frame")
                continue
            
            # Convert to base64 string for sending over Socket.IO
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # Send the frame to the web UI
            ui.send_message('webcam_frame', {
                'image': jpg_as_text,
                'timestamp': time.time()
            })
            
            # Calculate how long to sleep to maintain target fps
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_interval - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        if consecutive_failures >= max_failures:
            print("Error: Too many consecutive capture failures, stopping stream")
            ui.send_message('webcam_error', {
                'error': 'Camera stream lost'
            })
    
    def stop(self):
        """Stop the streaming thread and release the webcam."""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if self.capture:
            self.capture.release()
        
        print("Webcam streaming stopped")

# Initialize the Web UI
ui = WebUI()

# Create and start the webcam streamer
# Try index 1 first (Arduino's default), will fallback to 0 if needed
streamer = WebcamStreamer(
    fps=10,
    camera_index=1,
    target_width=416,
    target_height=416
)


if streamer.start():
    print("Webcam streamer initialized successfully")
else:
    print("Failed to initialize webcam streamer")

# Run the app
App.run()