FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app
COPY src/test_app.py .

# Set environment variables
ENV PORT=10000
ENV PYTHONUNBUFFERED=1

# Run the app
CMD gunicorn --bind 0.0.0.0:$PORT test_app:app 