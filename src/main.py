from webcam_capture import WebcamCapture
from object_detector import ObjectDetector
from display_manager import DisplayManager
import cv2
import time
from collections import deque

def main():
    # Initialize components with optimized settings
    detector = ObjectDetector(model_name="yolov8n.pt")  # Using nano model for speed
    display = DisplayManager()
    
    # Frame rate calculation variables
    fps_times = deque(maxlen=30)

    # Start webcam capture with reduced resolution
    with WebcamCapture(width=640, height=480) as webcam:
        if not webcam.start():
            print("Error: Could not start webcam capture")
            return

        print("Press 'q' to quit")
        print(f"Using device: {detector.device}")
        
        while True:
            loop_start = time.time()
            
            # Read frame
            success, frame = webcam.read_frame()
            if not success:
                print("Error: Could not read frame")
                break

            # Detect objects
            detections = detector.detect(frame)

            # Draw detections (only if there are any to save processing time)
            if detections:
                output_frame = frame.copy()
                for det in detections:
                    bbox = det['bbox'].astype(int)
                    label = det['label']
                    conf = det['confidence']
                    
                    # Draw bbox
                    cv2.rectangle(output_frame, 
                                (bbox[0], bbox[1]), 
                                (bbox[2], bbox[3]), 
                                (0, 255, 0), 2)
                    
                    # Draw label with confidence
                    text = f"{label} {conf:.2f}"
                    cv2.putText(output_frame, text,
                              (bbox[0], bbox[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (0, 255, 0), 2)
            else:
                output_frame = frame  # Use original frame if no detections
            
            # Calculate and display FPS
            fps_times.append(time.time() - loop_start)
            fps = int(1.0 / (sum(fps_times) / len(fps_times)))
            cv2.putText(output_frame, f"FPS: {fps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show frame
            display.show_frame(output_frame)

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    display.cleanup()

if __name__ == "__main__":
    main() 