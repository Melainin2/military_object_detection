FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies (pinned minimal set)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi==0.136.1 \
        uvicorn==0.46.0 \
        ultralytics==8.4.41 \
        opencv-python-headless==4.13.0.92 \
        python-multipart==0.0.26 \
        torch==2.11.0 \
        torchvision==0.26.0 \
        huggingface_hub>=0.23.0 \
        Pillow>=10.0.0

# Copy backend source
COPY . /app

EXPOSE 8000

# Run the FastAPI app (when built with backend as root directory, use main:app)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
