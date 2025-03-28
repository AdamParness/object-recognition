from ultralytics import YOLO
import numpy as np
import torch
import cv2
from typing import List, Dict, Optional
from threading import Lock
import logging
import os
from time import time

logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt"):
        """Initialize the YOLO object detector."""
        logger.info(f"Initializing ObjectDetector with model: {model_name}")
        
        # Force CPU for Render deployment
        self.device = 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Cache settings
        self.cache_duration = 0.1  # 100ms cache duration
        self.last_inference_time = 0
        self.cached_detections = None
        
        # Target size for inference (smaller = faster)
        self.target_width = 416  # Standard YOLO input size
        self.target_height = 416
            
        try:
            # Check if model exists
            if not os.path.exists(model_name):
                logger.error(f"Model file not found at {model_name}")
                raise FileNotFoundError(f"Model file not found at {model_name}")
                
            # Load model with error handling
            logger.info(f"Loading model from {model_name}")
            self.model = YOLO(model_name)
            
            # Set model to evaluation mode
            logger.info("Configuring model for inference")
            self.model.model.eval()
            torch.set_grad_enabled(False)
            
            # Enable model fusing for faster inference
            self.model.model.fuse()
            
            logger.info("YOLO model loaded successfully")
            
            # Store class names
            self.class_names = self.model.names
            logger.info(f"Model loaded with {len(self.class_names)} classes")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}", exc_info=True)
            raise
        
        self.lock = Lock()

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for inference."""
        # Calculate aspect ratio preserving resize
        height, width = frame.shape[:2]
        scale = min(self.target_width / width, self.target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create square image with padding
        square = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        x_offset = (self.target_width - new_width) // 2
        y_offset = (self.target_height - new_height) // 2
        square[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return square, (scale, x_offset, y_offset)

    def postprocess_detections(self, detections: List[Dict], original_shape: tuple, preprocess_params: tuple) -> List[Dict]:
        """Adjust detection coordinates back to original image size."""
        scale, x_offset, y_offset = preprocess_params
        height, width = original_shape[:2]
        
        adjusted_detections = []
        for det in detections:
            bbox = det['bbox']
            # Remove padding offset and scale back to original size
            x1 = max(0, min(width, (bbox[0] - x_offset) / scale))
            y1 = max(0, min(height, (bbox[1] - y_offset) / scale))
            x2 = max(0, min(width, (bbox[2] - x_offset) / scale))
            y2 = max(0, min(height, (bbox[3] - y_offset) / scale))
            
            adjusted_detections.append({
                'bbox': np.array([x1, y1, x2, y2]),
                'label': det['label'],
                'confidence': det['confidence']
            })
        
        return adjusted_detections

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in a frame."""
        try:
            current_time = time()
            
            # Check cache
            if (self.cached_detections is not None and 
                current_time - self.last_inference_time < self.cache_duration):
                return self.cached_detections
            
            with self.lock:
                # Ensure frame is in correct format (BGR)
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    logger.error(f"Invalid frame shape: {frame.shape}")
                    return []
                
                # Preprocess frame
                processed_frame, preprocess_params = self.preprocess_frame(frame)
                
                # Run inference
                logger.debug(f"Running inference on processed frame shape: {processed_frame.shape}")
                with torch.no_grad():
                    results = self.model.predict(
                        source=processed_frame,
                        conf=0.25,  # Slightly higher confidence for better performance
                        iou=0.45,   # NMS IoU threshold
                        max_det=10,  # Reduced for better performance
                        verbose=False,
                        device=self.device
                    )[0]
                
                # Process results
                detections = []
                
                if len(results.boxes) > 0:
                    boxes = results.boxes.cpu().numpy()
                    logger.debug(f"Found {len(boxes)} detections")
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        confidence = float(box.conf)
                        class_id = int(box.cls)
                        label = self.class_names[class_id]
                        
                        logger.debug(f"Detection: {label} with confidence {confidence:.2f}")
                        
                        detections.append({
                            'bbox': np.array([x1, y1, x2, y2]),
                            'label': label,
                            'confidence': confidence
                        })
                
                # Adjust detections back to original frame size
                detections = self.postprocess_detections(detections, frame.shape, preprocess_params)
                
                # Update cache
                self.cached_detections = detections
                self.last_inference_time = current_time
                
                return detections
                
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}", exc_info=True)
            return []