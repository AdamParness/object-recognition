from ultralytics import YOLO
import numpy as np
import torch
import cv2
from typing import List, Dict
from threading import Lock
import logging
import os

logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt"):
        """Initialize the YOLO object detector."""
        logger.info(f"Initializing ObjectDetector with model: {model_name}")
        
        # Auto-select device if none specified
        self.device = 'cpu'  # Force CPU for Render deployment
        logger.info(f"Using device: {self.device}")
            
        try:
            # Load model - use half precision for memory efficiency
            self.model = YOLO(model_name)
            self.model.model.half()  # Convert to half precision
            torch.set_grad_enabled(False)  # Disable gradients
            logger.info("YOLO model loaded successfully")
            
            # Store class names
            self.class_names = self.model.names
            logger.info(f"Model loaded with {len(self.class_names)} classes")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}", exc_info=True)
            raise
        
        self.lock = Lock()

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in a frame."""
        try:
            with self.lock:
                # Run inference with half precision
                logger.debug(f"Running inference on frame shape: {frame.shape}")
                results = self.model.predict(
                    source=frame,
                    conf=0.25,  # NMS confidence threshold
                    iou=0.45,   # NMS IoU threshold
                    max_det=20,  # Maximum number of detections per image
                    verbose=False,
                    half=True  # Use half precision
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
                else:
                    logger.debug("No detections found")
                
                return detections
                
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}", exc_info=True)
            return []