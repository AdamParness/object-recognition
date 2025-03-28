FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY src/ src/

# Add src directory to Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Set environment variables
ENV PORT=10000
ENV PYTHONUNBUFFERED=1
ENV YOLO_MODEL=yolov8n.pt

# Run the application
CMD gunicorn --worker-class eventlet -w 1 \
    --log-level debug \
    --error-logfile - \
    --access-logfile - \
    --bind 0.0.0.0:$PORT \
    --chdir /app/src \
    app:app 