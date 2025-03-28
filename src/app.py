from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from object_detector import ObjectDetector
import json
import socket
import sys
import signal
import os
from threading import Lock
import logging
from werkzeug.middleware.proxy_fix import ProxyFix

# Set up logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# Configure SocketIO with more lenient settings for Render
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1e8,
    manage_session=False
)

# Initialize the object detector
try:
    logger.info("Initializing object detector...")
    detector = ObjectDetector(model_name="yolov8n.pt")
    logger.info(f"Object detector initialized successfully. Using device: {detector.device}")
except Exception as e:
    logger.error(f"Error initializing object detector: {str(e)}")
    sys.exit(1)

# Add thread lock for thread-safe processing
thread_lock = Lock()

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\nShutting down server...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

@app.route('/')
def index():
    """Serve the index HTML page."""
    logger.info("Serving index page")
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info('Client connected')
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info('Client disconnected')

@socketio.on('frame')
def handle_frame(data):
    """Process received frame and send back detections."""
    try:
        with thread_lock:
            # Decode base64 image
            try:
                encoded_data = data.split(',')[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.error("Failed to decode frame")
                    return
                    
                logger.debug(f"Decoded frame shape: {frame.shape}")
            except Exception as e:
                logger.error(f"Error decoding frame: {str(e)}")
                return

            # Get detections
            detections = detector.detect(frame)
            
            # Log detections
            if detections:
                logger.info(f"Found {len(detections)} detections:")
                for det in detections:
                    logger.info(f"  - {det['label']} (confidence: {det['confidence']:.3f}, bbox: {det['bbox'].tolist()})")
            else:
                logger.debug("No detections in this frame")
            
            # Convert detections to serializable format
            detection_list = []
            for det in detections:
                # Ensure bbox coordinates are within frame boundaries
                bbox = det['bbox']
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = min(frame.shape[1], bbox[2])
                bbox[3] = min(frame.shape[0], bbox[3])
                
                detection_list.append({
                    'bbox': bbox.tolist(),
                    'label': det['label'],
                    'confidence': float(det['confidence'])
                })

            # Send back the detections
            emit('detections', {'detections': detection_list})
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}", exc_info=True)

def find_free_port(start_port=8000, max_port=9000):
    """Find a free port between start_port and max_port."""
    for port in range(start_port, max_port):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', port))
            s.close()
            return port
        except OSError:
            continue
    return None

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 10000))
        host = os.environ.get('HOST', '0.0.0.0')
        
        logger.info(f"Starting server on {host}:{port}")
        socketio.run(app,
                    host=host,
                    port=port,
                    debug=False,
                    allow_unsafe_werkzeug=True,
                    log_output=True)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}", exc_info=True)
        sys.exit(1) 