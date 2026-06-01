import logging
import os
import uuid
import gc
import time
import threading

import torch

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2

# ===============================
# PyTorch: single thread for 0.1 CPU
# ===============================
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ===============================
# Logging
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backend")

# ===============================
# App
# ===============================
app = FastAPI(title="AI Detection Dashboard", version="2.0.0")

# ===============================
# CORS
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
# Constants
# ===============================
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
MAX_INFERENCE_SIZE = 320
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
YOLO_IMGSZ = 320
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
# Model loading (thread-safe, once)
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


# ===============================
# Startup / Shutdown
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
    cv2.destroyAllWindows()


# ===============================
# Routes
# ===============================
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model_loaded}


@app.get("/")
def home():
    return FileResponse("frontend/index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...), request: Request = None):
    req_start = time.time()
    upload_path = None

    try:
        client_ip = request.client.host if request and request.client else "unknown"
        filename = os.path.basename(file.filename or "")
        logger.info("=== PREDICT from %s — file: %s ===", client_ip, filename)

        # --- Validation ---
        if os.path.splitext(filename.lower())[1] not in ALLOWED_EXTENSIONS:
            logger.warning("REJECTED extension: %s", filename)
            return JSONResponse({"error": "Only image files (JPG, JPEG, PNG, WEBP) are allowed."}, status_code=400)

        if file.content_type not in ALLOWED_MIME_TYPES:
            logger.warning("REJECTED MIME: %s = %s", filename, file.content_type)
            return JSONResponse({"error": "Only image files (JPG, JPEG, PNG, WEBP) are allowed."}, status_code=400)

        ext = filename.rsplit(".", 1)[-1].lower()
        unique_name = f"{uuid.uuid4()}.{ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_name)
        upload_path = file_path

        # --- Upload ---
        t_upload = time.time()
        size = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(8192):
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE:
                    logger.warning("REJECTED too large: %d bytes from %s", size, client_ip)
                    return JSONResponse({"error": "File too large (max 10 MB)"}, status_code=413)
                buffer.write(chunk)
        upload_elapsed = time.time() - t_upload
        logger.info("timing: upload_read=%.2fs size=%d", upload_elapsed, size)

        # --- Request trace ---
        logger.info("TRACE: request_received client=%s file=%s size=%d", client_ip, filename, size)
        logger.info("TRACE: config imgsz=%d conf=%.2f iou=%.2f max_inference_size=%d", YOLO_IMGSZ, 0.25, 0.5, MAX_INFERENCE_SIZE)

        # --- Image decode + resize ---
        t_img = time.time()
        img = cv2.imread(file_path)
        if img is None:
            logger.error("cv2.imread failed for %s (size=%d)", file_path, size)
            return JSONResponse({"error": "Invalid image file"}, status_code=400)

        h, w = img.shape[:2]
        orig_size = f"{w}x{h}"
        if max(h, w) > MAX_INFERENCE_SIZE:
            scale = MAX_INFERENCE_SIZE / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            logger.info("timing: resized %s -> %dx%d", orig_size, new_w, new_h)
        logger.info("timing: preprocess=%.2fs img=%s", time.time() - t_img, str(img.shape))

        # --- Ensure model loaded ---
        t_model = time.time()
        loaded_model = load_model_sync()
        logger.info("timing: model_ensure=%.2fs (loaded=%s)", time.time() - t_model, model_loaded)
        logger.info("TRACE: model_path=%s", loaded_model.ckpt_path if hasattr(loaded_model, 'ckpt_path') else 'unknown')

        # --- GC before inference ---
        gc.collect()

        # --- Inference (imgsz=320 for speed on 0.1 CPU) ---
        t_infer = time.time()
        logger.info("TRACE: inference_start img_shape=(%d,%d,%d)", img.shape[0], img.shape[1], img.shape[2])
        try:
            with torch.no_grad():
                results = loaded_model(
                    img,
                    imgsz=YOLO_IMGSZ,
                    conf=0.25,
                    iou=0.5,
                    device="cpu",
                    verbose=False,
                )
        except Exception as infer_err:
            logger.exception("Inference crashed after %.2fs", time.time() - t_infer)
            return JSONResponse({"error": "Model inference failed"}, status_code=500)

        inference_elapsed = time.time() - t_infer
        logger.info("timing: inference=%.2fs", inference_elapsed)
        logger.info("TRACE: inference_end duration=%.3f", inference_elapsed)

        # --- Post-process ---
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
                    logger.info("DETECTED: %s conf=%.4f", class_name, conf)

        # --- Save output ---
        output_path = os.path.join(OUTPUT_DIR, unique_name)
        cv2.imwrite(output_path, img)
        render_elapsed = time.time() - t_render
        logger.info("timing: render=%.2fs detections=%d", render_elapsed, len(detections))

        total_elapsed = time.time() - req_start
        logger.info(
            "timing: total=%.2fs inference=%.2fs detections=%d img=%s client=%s",
            total_elapsed, inference_elapsed, len(detections), orig_size, client_ip,
        )

        # --- Cleanup ---
        del img, results
        loaded_model.predictor = None
        gc.collect()
        gc.collect()
        gc.collect()

        if not military_found:
            logger.info("RESULT: No military objects in %s", unique_name)
            logger.info("TRACE: response_sent status=200 detections=0 elapsed=%.2fs", total_elapsed)
            return JSONResponse({
                "message": "No military object detected",
                "detections": [],
                "image_url": f"/outputs/{unique_name}",
            })

        logger.info("RESULT: %d military objects in %s", len(detections), unique_name)
        logger.info("TRACE: response_sent status=200 detections=%d elapsed=%.2fs", len(detections), total_elapsed)
        return JSONResponse({
            "message": "Military objects detected",
            "detections": detections,
            "image_url": f"/outputs/{unique_name}",
        })

    except Exception as e:
        elapsed = time.time() - req_start
        logger.exception("FAILED after %.2fs — %s: %s", elapsed, type(e).__name__, str(e))
        return JSONResponse({"error": "Internal server error"}, status_code=500)
    finally:
        if upload_path and os.path.exists(upload_path):
            os.remove(upload_path)
            logger.info("CLEANUP: removed %s", upload_path)


# ===============================
# Static files
# ===============================
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
