import logging
import os
import uuid
import gc
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2

# ===============================
# PyTorch thread optimization for single-CPU environments
# ===============================
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
app = FastAPI(title="AI Detection Dashboard", version="2.0.0")

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
# Directories & Constants
# ===============================
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
MAX_INFERENCE_SIZE = 640
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
INFERENCE_TIMEOUT = 300
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# Global state
# ===============================
model = None
model_loaded = False
model_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)

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
# Model Loading (thread-safe, loads once)
# ===============================
def load_model_sync():
    global model, model_loaded

    if model is not None:
        return model

    with model_lock:
        if model is not None:
            return model

        logger.info("[MODEL] Downloading model from Hugging Face Hub...")
        t0 = time.time()
        model_path = hf_hub_download(
            repo_id="datasidahmed/military_object_detection",
            filename="best.pt",
            token=os.getenv("HF_TOKEN"),
        )
        logger.info("[MODEL] Downloaded in %.2fs: %s", time.time() - t0, model_path)

        logger.info("[MODEL] Loading model into memory...")
        t1 = time.time()
        model = YOLO(model_path)
        model_loaded = True
        logger.info("[MODEL] Loaded in %.2fs — ready for inference", time.time() - t1)
        return model


async def ensure_model():
    """Ensure model is loaded. If a background load is in progress, waits for it."""
    if model is not None:
        return model
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, load_model_sync)


# ===============================
# Startup / Shutdown Events
# ===============================
@app.on_event("startup")
def on_startup():
    logger.info("=== AI Detection Dashboard starting up ===")
    logger.info("Upload and output directories ready")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN is not set — model download may fail for gated models")
    else:
        logger.info("HF_TOKEN is configured")

    logger.info("[MODEL] Starting background model pre-load...")
    thread = threading.Thread(target=load_model_sync, daemon=True)
    thread.start()

    logger.info("=== Startup complete — model pre-load in background ===")


@app.on_event("shutdown")
def on_shutdown():
    logger.info("Shutting down AI Detection Dashboard")
    executor.shutdown(wait=False)
    cv2.destroyAllWindows()


# ===============================
# Health Check
# ===============================
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model_loaded}


# ===============================
# Home Route
# ===============================
@app.get("/")
def home():
    return FileResponse("frontend/index.html")


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
        logger.info("=== PREDICT REQUEST from %s — file: %s ===", client_ip, filename)

        # --- Step 1: File validation ---
        if os.path.splitext(filename.lower())[1] not in ALLOWED_EXTENSIONS:
            logger.warning("REJECTED — invalid extension: %s", filename)
            return JSONResponse({"error": "Only image files (JPG, JPEG, PNG, WEBP) are allowed."}, status_code=400)

        if file.content_type not in ALLOWED_MIME_TYPES:
            logger.warning("REJECTED — invalid MIME: %s = %s", filename, file.content_type)
            return JSONResponse({"error": "Only image files (JPG, JPEG, PNG, WEBP) are allowed."}, status_code=400)

        ext = filename.rsplit(".", 1)[-1].lower()
        unique_name = f"{uuid.uuid4()}.{ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_name)
        upload_path = file_path

        # --- Step 2: Upload read ---
        t_upload = time.time()
        size = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(8192):
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE:
                    logger.warning("REJECTED — file too large: %d bytes from %s", size, client_ip)
                    return JSONResponse({"error": "File too large (max 10 MB)"}, status_code=413)
                buffer.write(chunk)
        upload_elapsed = time.time() - t_upload
        logger.info("timing: upload_read=%.2fs size=%d bytes", upload_elapsed, size)

        # --- Step 3: Image loading ---
        t_img = time.time()
        img = cv2.imread(file_path)
        if img is None:
            logger.error("FAILED — cv2.imread returned None for %s (size=%d)", file_path, size)
            return JSONResponse({"error": "Invalid image file"}, status_code=400)

        h, w = img.shape[:2]
        orig_size = f"{w}x{h}"
        logger.info("timing: image_load=%.2fs dimensions=%s channels=%d", time.time() - t_img, orig_size, img.shape[2] if len(img.shape) > 2 else 1)

        # --- Step 4: Image resizing ---
        t_resize = time.time()
        if max(h, w) > MAX_INFERENCE_SIZE:
            scale = MAX_INFERENCE_SIZE / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            logger.info("timing: resized %s -> %dx%d in %.2fs", orig_size, new_w, new_h, time.time() - t_resize)
        else:
            logger.info("timing: no_resize_needed %s (max dim <= %d)", orig_size, MAX_INFERENCE_SIZE)

        # --- Step 5: Model loading check ---
        t_model = time.time()
        loaded_model = await ensure_model()
        model_check_elapsed = time.time() - t_model
        logger.info("timing: model_ensure=%.2fs (already_loaded=%s)", model_check_elapsed, model_loaded)

        # --- Step 6: YOLO inference ---
        t_infer = time.time()
        loop = asyncio.get_event_loop()
        try:
            results = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    lambda: loaded_model(img, conf=0.25, iou=0.5, device="cpu"),
                ),
                timeout=INFERENCE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("INFERENCE TIMEOUT after %ds — image=%s client=%s", INFERENCE_TIMEOUT, orig_size, client_ip)
            return JSONResponse(
                {"error": "Inference timed out — the free hosting tier is too slow for this image. Try a smaller image."},
                status_code=504,
            )

        inference_elapsed = time.time() - t_infer
        logger.info("timing: inference=%.2fs", inference_elapsed)

        # --- Step 7: Result rendering ---
        t_render = time.time()
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
                    logger.info("DETECTED: %s conf=%.4f box=[%d,%d,%d,%d]", class_name, conf, x1, y1, x2, y2)

        # --- Step 8: Save output + response ---
        output_path = os.path.join(OUTPUT_DIR, unique_name)
        cv2.imwrite(output_path, img)
        render_elapsed = time.time() - t_render
        logger.info("timing: render=%.2fs detections=%d military=%s output=%s", render_elapsed, len(detections), military_found, unique_name)

        total_elapsed = time.time() - req_start
        logger.info(
            "timing: total=%.2fs breakdown=[upload=%.2fs img=%.2fs resize=%.2fs model=%.2fs inference=%.2fs render=%.2fs] detections=%d img=%s client=%s",
            total_elapsed,
            upload_elapsed,
            time.time() - t_img - (time.time() - t_resize) if max(h, w) > MAX_INFERENCE_SIZE else 0,
            time.time() - t_resize if max(h, w) > MAX_INFERENCE_SIZE else 0,
            model_check_elapsed,
            inference_elapsed,
            render_elapsed,
            len(detections),
            orig_size,
            client_ip,
        )

        # --- Memory cleanup ---
        del img, results
        gc.collect()

        if not military_found:
            logger.info("RESULT: No military objects detected in %s from %s", unique_name, client_ip)
            return JSONResponse({
                "message": "No military object detected",
                "detections": [],
                "image_url": f"/outputs/{unique_name}",
            })

        logger.info("RESULT: %d military objects detected in %s from %s", len(detections), unique_name, client_ip)
        return JSONResponse({
            "message": "Military objects detected",
            "detections": detections,
            "image_url": f"/outputs/{unique_name}",
        })

    except Exception as e:
        elapsed = time.time() - req_start
        logger.exception("UNHANDLED EXCEPTION after %.2fs — %s: %s", elapsed, type(e).__name__, str(e))
        return JSONResponse({"error": "Internal server error"}, status_code=500)
    finally:
        if upload_path and os.path.exists(upload_path):
            os.remove(upload_path)
            logger.info("CLEANUP: removed upload %s", upload_path)


# ===============================
# Static Files
# ===============================
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
