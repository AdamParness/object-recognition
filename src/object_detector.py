from ultralytics import YOLO
import numpy as np
import torch
import cv2
from typing import List, Dict, Set
from threading import Lock

class ObjectDetector:
    # Available models with their characteristics
    AVAILABLE_MODELS = {
        'yolov8n': {'name': 'yolov8n.pt', 'desc': 'Nano - Fastest, lowest accuracy'},
        'yolov8s': {'name': 'yolov8s.pt', 'desc': 'Small - Better accuracy, good speed'},
        'yolov8m': {'name': 'yolov8m.pt', 'desc': 'Medium - Balanced accuracy/speed'},
        'yolov8l': {'name': 'yolov8l.pt', 'desc': 'Large - High accuracy, slower'},
        'yolov8x': {'name': 'yolov8x.pt', 'desc': 'XLarge - Highest accuracy, slowest'}
    }

    def __init__(self, model_name: str = "yolov8n.pt", device: str = None):
        """
        Initialize the YOLO object detector.
        
        Args:
            model_name (str): Name of the YOLO model to use
            device (str): Device to run the model on ('cpu', 'cuda', or None for auto)
        """
        # Auto-select device if none specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load model with optimization settings
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        # Enable model optimizations
        if self.device == 'cuda':
            self.model.model.half()  # FP16 for GPU
        else:
            # CPU optimizations
            torch.set_num_threads(4)  # Limit CPU threads
            if hasattr(torch, 'set_num_interop_threads'):
                torch.set_num_interop_threads(1)
        
        self.frame_count = 0
        self.last_detections = []
        self.lock = Lock()
        
        # Pre-define target size for faster processing
        self.target_size = (320, 320)  # Reduced resolution
        self.frame_skip = 2  # Process every 2nd frame

    @classmethod
    def list_available_models(cls):
        """List all available models with their descriptions."""
        print("\nAvailable YOLO Models:")
        print("-" * 50)
        for key, value in cls.AVAILABLE_MODELS.items():
            print(f"{key}: {value['desc']}")
        print("-" * 50)

    def _classify_region(self, frame: np.ndarray, bbox: np.ndarray) -> str:
        """Classify a specific region using the classification model."""
        x1, y1, x2, y2 = map(int, bbox)
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return None
            
        # Ensure minimum size and proper aspect ratio
        min_size = 32
        if region.shape[0] < min_size or region.shape[1] < min_size:
            return None
            
        # Classify the region
        results = self.classifier(region, verbose=False)[0]
        if results.probs is None:
            return None
            
        label = results.names[results.probs.top1]
        return label

    def _post_process_detections(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """Apply post-processing to improve detection accuracy."""
        processed = []
        
        for det in detections:
            label = det['label']
            conf = det['confidence']
            bbox = det['bbox']
            
            # Additional classification for small objects
            if any(label in category for category in self.categories.values()):
                classified_label = self._classify_region(frame, bbox)
                if classified_label:
                    # If classification confidence is high, update the label
                    det['label'] = classified_label
            
            # Apply size-based filtering
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            frame_area = frame.shape[0] * frame.shape[1]
            
            # Filter out unlikely detections based on size
            if label in self.categories['small_objects'] and area > 0.3 * frame_area:
                continue
            if label in self.categories['furniture'] and area < 0.01 * frame_area:
                continue
                
            processed.append(det)
            
        return processed

    def _temporal_smoothing(self, detections: List[Dict]) -> List[Dict]:
        """Apply temporal smoothing to reduce false detections."""
        self.detection_history.append(detections)
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
            
        # Only keep detections that appear in majority of recent frames
        smoothed = []
        for det in detections:
            label = det['label']
            bbox = det['bbox']
            
            # Count similar detections in history
            count = 1
            for hist_dets in self.detection_history[:-1]:
                for hist_det in hist_dets:
                    if (hist_det['label'] == label and 
                        self._iou(bbox, hist_det['bbox']) > 0.3):
                        count += 1
                        break
                        
            # Add detection if it appears in majority of frames
            if count >= len(self.detection_history) // 2:
                smoothed.append(det)
                
        return smoothed

    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.3) -> List[Dict]:
        """
        Detect objects in a frame.
        
        Args:
            frame (np.ndarray): Input frame
            conf_threshold (float): Confidence threshold for detections
            
        Returns:
            List[Dict]: List of detections with bounding boxes, labels, and confidence scores
        """
        self.frame_count += 1
        
        # Skip frames for better performance
        if self.frame_count % self.frame_skip != 0:
            return self.last_detections
            
        # Fast resize without aspect ratio preservation
        frame_resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Calculate scale factors for bbox conversion
        scale_x = frame.shape[1] / self.target_size[0]
        scale_y = frame.shape[0] / self.target_size[1]
        
        with self.lock:
            # Run inference with optimized settings
            results = self.model(frame_resized, 
                               verbose=False,
                               conf=conf_threshold,
                               iou=0.45,
                               agnostic_nms=True,  # Class-agnostic NMS
                               max_det=5)[0]  # Limit detections
            
            detections = []
            for box in results.boxes:
                confidence = float(box.conf)
                if confidence < conf_threshold:
                    continue

                label = results.names[int(box.cls)]
                bbox = box.xyxy[0].cpu().numpy()
                
                # Scale bbox back to original size
                bbox[0] *= scale_x
                bbox[1] *= scale_y
                bbox[2] *= scale_x
                bbox[3] *= scale_y
                
                detection = {
                    'bbox': bbox,
                    'label': label,
                    'confidence': confidence
                }
                detections.append(detection)

            self.last_detections = detections
            return detections