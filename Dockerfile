FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/test_app.py .

ENV PORT=10000
ENV PYTHONUNBUFFERED=1

CMD gunicorn --bind 0.0.0.0:$PORT test_app:app
