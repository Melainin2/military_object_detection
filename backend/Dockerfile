FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies (pinned minimal set)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi==0.136.1 \
        uvicorn==0.46.0 \
        onnxruntime \
        opencv-python-headless==4.13.0.92 \
        python-multipart==0.0.26 \
        huggingface_hub>=0.23.0 \
        numpy>=1.24.0 \
        Pillow>=10.0.0 \
        torch \
        torchvision

# Copy backend source
COPY . /app

EXPOSE 8000

# Run the FastAPI app (when built with backend as root directory)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
