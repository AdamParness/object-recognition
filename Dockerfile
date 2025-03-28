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

# Make startup script executable
RUN chmod +x start.sh

# Create necessary directories
RUN mkdir -p /root/.cache/torch/hub/checkpoints/

# Set environment variables
ENV PORT=10000
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV YOLO_MODEL=yolov8n.pt

# Health check using our new health endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Use our startup script
CMD ["./start.sh"] 