import cv2
import numpy as np
from typing import List, Dict

class DisplayManager:
    def __init__(self):
        """Initialize the display manager with default settings."""
        self.colors = np.random.uniform(0, 255, size=(80, 3))  # Random colors for different classes

    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            detections (List[Dict]): List of detections with bounding boxes and labels
            
        Returns:
            np.ndarray: Frame with drawn detections
        """
        output_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox'].astype(int)
            label = detection['label']
            confidence = detection['confidence']
            
            # Get color for this class
            color = self.colors[hash(label) % len(self.colors)]
            color = tuple(map(int, color))
            
            # Draw bounding box
            cv2.rectangle(output_frame, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color, 2)
            
            # Draw label and confidence
            label_text = f'{label}: {confidence:.2f}'
            cv2.putText(output_frame, 
                       label_text, 
                       (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       color, 
                       2)
        
        return output_frame

    def show_frame(self, frame: np.ndarray, window_name: str = "Object Detection"):
        """
        Display a frame in a window.
        
        Args:
            frame (np.ndarray): Frame to display
            window_name (str): Name of the window
        """
        cv2.imshow(window_name, frame)

    def cleanup(self):
        """Clean up OpenCV windows."""
        cv2.destroyAllWindows() 