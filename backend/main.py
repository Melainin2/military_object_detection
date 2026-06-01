import logging
import shutil
import os
import uuid
import gc
import time

import torch

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2

# PyTorch thread optimization for single-CPU environments
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ===============================
# Logging Configuration
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backend")

# ===============================
# Application
# ===============================
app = FastAPI(title="AI Detection Dashboard", version="1.0.0")

# ===============================
# CORS Middleware
# ===============================
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Directories
# ===============================
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
MAX_INFERENCE_SIZE = 640
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# Model (lazy loaded)
# ===============================
model = None

# ===============================
# Class definitions
# ===============================
military_classes = [
    "camouflage_soldier", "weapon", "military_tank",
    "military_truck", "military_vehicle", "soldier",
    "artillery", "military_aircraft", "warship",
]

class_names = [
    "camouflage_soldier", "weapon", "military_tank",
    "military_truck", "military_vehicle", "civilian",
    "soldier", "civilian_vehicle", "artillery",
    "military_aircraft", "warship",
]


# ===============================
# Startup / Shutdown Events
# ===============================
@app.on_event("startup")
def on_startup():
    logger.info("Starting AI Detection Dashboard...")
    logger.info("Upload and output directories ready")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN is not set - model download may fail for gated models")
    else:
        logger.info("HF_TOKEN is configured")

    logger.info("Startup complete - ready for requests")


@app.on_event("shutdown")
def on_shutdown():
    logger.info("Shutting down AI Detection Dashboard")
    cv2.destroyAllWindows()


# ===============================
# Health Check
# ===============================
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


# ===============================
# Home Route
# ===============================
@app.get("/")
def home():
    return FileResponse("frontend/index.html")


# ===============================
# Model Loading
# ===============================
def load_model():
    global model

    if model is not None:
        return model

    logger.info("Downloading model from Hugging Face Hub...")
    model_path = hf_hub_download(
        repo_id="datasidahmed/military_object_detection",
        filename="best.pt",
        token=os.getenv("HF_TOKEN"),
    )
    logger.info("Model downloaded: %s", model_path)

    logger.info("Loading model into memory...")
    model = YOLO(model_path)
    logger.info("Model loaded successfully")
    return model


# ===============================
# Prediction Route
# ===============================
@app.post("/predict")
async def predict(file: UploadFile = File(...), request: Request = None):
    req_start = time.time()
    upload_path = None

    try:
        client_ip = request.client.host if request and request.client else "unknown"
        filename = os.path.basename(file.filename or "")
        logger.info("Predict request from %s - file: %s", client_ip, filename)

        if os.path.splitext(filename.lower())[1] not in ALLOWED_EXTENSIONS:
            return JSONResponse({"error": "Only image files (JPG, JPEG, PNG, WEBP) are allowed."}, status_code=400)

        if file.content_type not in ALLOWED_MIME_TYPES:
            return JSONResponse({"error": "Only image files (JPG, JPEG, PNG, WEBP) are allowed."}, status_code=400)

        ext = filename.rsplit(".", 1)[-1].lower()
        unique_name = f"{uuid.uuid4()}.{ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_name)
        upload_path = file_path

        t0 = time.time()
        size = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(8192):
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE:
                    return JSONResponse({"error": "File too large (max 10 MB)"}, status_code=413)
                buffer.write(chunk)
        logger.info("timing: upload_read=%.2fs size=%d", time.time() - t0, size)

        t1 = time.time()
        img = cv2.imread(file_path)
        if img is None:
            logger.error("cv2.imread failed for %s", file_path)
            return JSONResponse({"error": "Invalid image file"}, status_code=400)

        h, w = img.shape[:2]
        orig_size = f"{w}x{h}"
        if max(h, w) > MAX_INFERENCE_SIZE:
            scale = MAX_INFERENCE_SIZE / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            logger.info("timing: resized %s -> %dx%d", orig_size, new_w, new_h)
        logger.info("timing: preprocess=%.2fs img_shape=%s", time.time() - t1, str(img.shape))

        t2 = time.time()
        loaded_model = load_model()
        logger.info("timing: model_load_check=%.2fs", time.time() - t2)

        t3 = time.time()
        with torch.no_grad():
            results = loaded_model(img, conf=0.25, iou=0.5, device="cpu")
        inference_time = time.time() - t3
        logger.info("timing: inference=%.2fs", inference_time)

        t4 = time.time()
        detections = []
        military_found = False
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = class_names[cls_id] if cls_id < len(class_names) else "unknown"
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if class_name in military_classes:
                    military_found = True
                    detections.append({
                        "class_name": class_name,
                        "confidence": round(conf, 4),
                        "box": [x1, y1, x2, y2],
                    })
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        img, f"{class_name} {conf:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                    )

        output_path = os.path.join(OUTPUT_DIR, unique_name)
        cv2.imwrite(output_path, img)
        logger.info("timing: postprocess=%.2fs", time.time() - t4)

        total = time.time() - req_start
        logger.info(
            "timing: total=%.2fs inference=%.2fs detections=%d img=%s client=%s",
            total, inference_time, len(detections), orig_size, client_ip,
        )

        # Memory cleanup
        del img, results
        gc.collect()

        if not military_found:
            return JSONResponse({
                "message": "No military object detected",
                "detections": [],
                "image_url": f"/outputs/{unique_name}",
            })

        return JSONResponse({
            "message": "Military objects detected",
            "detections": detections,
            "image_url": f"/outputs/{unique_name}",
        })

    except Exception as e:
        elapsed = time.time() - req_start
        logger.exception("Prediction failed after %.2fs", elapsed)
        return JSONResponse({"error": "Internal server error"}, status_code=500)
    finally:
        if upload_path and os.path.exists(upload_path):
            os.remove(upload_path)
            logger.info("Cleaned up upload: %s", upload_path)


# ===============================
# Static Files
# ===============================
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
