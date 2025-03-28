#!/bin/bash

# Print environment information
echo "Starting application..."
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "Contents of directory:"
ls -la

# Try to download the model first
echo "Attempting to download YOLO model..."
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Model loaded successfully')" || exit 1

# Start the application
echo "Starting Gunicorn server..."
exec gunicorn \
    --worker-class eventlet \
    --workers 1 \
    --log-level debug \
    --error-logfile - \
    --access-logfile - \
    --bind 0.0.0.0:${PORT} \
    --timeout 120 \
    --graceful-timeout 60 \
    --capture-output \
    src.app:app 