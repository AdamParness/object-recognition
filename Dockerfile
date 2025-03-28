FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies with verbose output and error handling
RUN pip install --no-cache-dir -r requirements.txt 2>&1 | tee pip_install.log || (cat pip_install.log && exit 1)

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /root/.cache/torch/hub/checkpoints/

# Download YOLO model during build with error handling
RUN python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Model loaded successfully')" 2>&1 | tee yolo_download.log || (cat yolo_download.log && exit 1)

# Set environment variables
ENV PORT=10000
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV YOLO_MODEL=yolov8n.pt

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

# Command to run the application
CMD gunicorn \
    --worker-class eventlet \
    --workers 1 \
    --log-level debug \
    --error-logfile - \
    --access-logfile - \
    --bind 0.0.0.0:${PORT} \
    --timeout 120 \
    --graceful-timeout 60 \
    src.app:app 