FROM python:3.10-slim

WORKDIR /app

# Install system dependencies more efficiently
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies efficiently
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip/*

# Copy application code
COPY . .

# Download YOLO model during build and clean up
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
    && rm -rf ~/.cache/torch/ \
    && rm -rf ~/.cache/pip/

# Set environment variables
ENV PORT=8000
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV YOLO_MODEL=yolov8n.pt

# Command to run the application
CMD gunicorn --worker-class eventlet -w 1 -b 0.0.0.0:${PORT} src.app:app 