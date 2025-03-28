import cv2
import numpy as np
from typing import Optional, Tuple

class WebcamCapture:
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """
        Initialize the webcam capture.
        
        Args:
            camera_id (int): ID of the camera to use (default is 0 for built-in webcam)
            width (int): Desired frame width
            height (int): Desired frame height
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None

    def start(self) -> bool:
        """Start the webcam capture."""
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_ANY)  # Use any available API
        
        if not self.cap.isOpened():
            return False
            
        # Set optimized capture properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set optimized format and encoding
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Disable automatic conversion
        
        # Additional performance settings
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure off
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -7)  # Set fixed exposure
        
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the webcam.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: Success flag and the captured frame
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
            
        ret, frame = self.cap.read()
        if not ret:
            return False, None
            
        # Convert frame format if needed (faster than automatic conversion)
        if len(frame.shape) == 2:  # If grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        return True, frame

    def release(self):
        """Release the webcam capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        """Context manager entry."""
        if not self.start():
            raise RuntimeError("Failed to start webcam capture")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release() 