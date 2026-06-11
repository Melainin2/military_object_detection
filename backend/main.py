import logging
import os
import uuid
import gc
import time
import threading

import numpy
import torch

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backend")

app = FastAPI(title="AI Detection Dashboard", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR     = "uploads"
OUTPUT_DIR     = "outputs"
YOLO_IMGSZ     = 320   # 320 keeps RAM under 512 MB on Render free tier
MAX_IMG_DIM    = 320   # resize before inference to same budget
YOLO_CONF      = 0.25
YOLO_IOU       = 0.40
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

HF_REPO_ID = "datasidahmed/YOLOV8"
HF_FILENAME = "best.pt"
MODEL_CACHE_PATH = "best.pt"

model = None
model_loaded = False
model_lock = threading.Lock()

class_names = [
    "camouflage_soldier",   # 0
    "weapon",               # 1
    "military_tank",        # 2
    "military_truck",       # 3
    "military_vehicle",     # 4
    "civilian",             # 5
    "soldier",              # 6
    "civilian_vehicle",     # 7
    "military_artillery",   # 8
    "trench",               # 9
    "military_aircraft",    # 10
    "military_warship",     # 11
]

military_classes = {
    "camouflage_soldier", "weapon", "military_tank", "military_truck",
    "military_vehicle", "soldier", "military_artillery", "trench",
    "military_aircraft", "military_warship",
}


def load_model_from_hf() -> YOLO:
    """Download weights from HuggingFace once, cache locally, keep in memory."""
    global model, model_loaded

    if model is not None:
        return model

    with model_lock:
        if model is not None:
            return model

        if os.path.exists(MODEL_CACHE_PATH):
            logger.info("[MODEL] Using cached weights: %s", MODEL_CACHE_PATH)
            model_path = MODEL_CACHE_PATH
        else:
            logger.info("[MODEL] Downloading from HuggingFace: %s / %s", HF_REPO_ID, HF_FILENAME)
            t0 = time.time()
            try:
                model_path = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=HF_FILENAME,
                    local_dir=".",
                    local_dir_use_symlinks=False,
                )
                logger.info("[MODEL] Downloaded in %.2fs", time.time() - t0)
            except Exception as exc:
                logger.error("[MODEL] HuggingFace download failed: %s", exc)
                raise RuntimeError(
                    f"Could not download model from HuggingFace ({HF_REPO_ID}): {exc}"
                ) from exc

        logger.info("[MODEL] Loading into memory...")
        t1 = time.time()
        model = YOLO(model_path)
        model_loaded = True
        logger.info("[MODEL] Ready in %.2fs", time.time() - t1)
        return model


def _warmup():
    dummy = numpy.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=numpy.uint8)
    with torch.no_grad():
        model.predict(dummy, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU, device="cpu", verbose=False)
    logger.info("[MODEL] Warmup complete — ready for inference")


def _startup_task():
    try:
        load_model_from_hf()
        _warmup()
    except Exception:
        logger.exception("[MODEL] Startup load failed — first request will retry")


@app.on_event("startup")
def on_startup():
    logger.info("=== AI Detection Dashboard v3 starting up ===")
    threading.Thread(target=_startup_task, daemon=True).start()


@app.on_event("shutdown")
def on_shutdown():
    cv2.destroyAllWindows()
    logger.info("Shutdown complete")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_source": f"huggingface.co/{HF_REPO_ID}",
    }


@app.get("/")
def home():
    return FileResponse("frontend/index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...), request: Request = None):
    req_start = time.time()
    upload_path = None

    try:
        filename = os.path.basename(file.filename or "")
        ext = os.path.splitext(filename.lower())[1]

        if ext not in ALLOWED_EXTENSIONS:
            return JSONResponse(
                {"error": "Only image files (JPG, JPEG, PNG, WEBP) are allowed."},
                status_code=400,
            )
        if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
            return JSONResponse(
                {"error": "Only image files (JPG, JPEG, PNG, WEBP) are allowed."},
                status_code=400,
            )

        # Load model (cached after first call)
        try:
            yolo = load_model_from_hf()
        except RuntimeError as exc:
            logger.error("[PREDICT] Model unavailable: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=503)

        # Save upload
        unique_name = f"{uuid.uuid4()}{ext}"
        upload_path = os.path.join(UPLOAD_DIR, unique_name)
        with open(upload_path, "wb") as buf:
            while chunk := await file.read(8192):
                buf.write(chunk)

        # Decode image
        img = cv2.imread(upload_path)
        if img is None:
            return JSONResponse({"error": "Invalid or unreadable image file."}, status_code=400)

        orig_h, orig_w = img.shape[:2]

        # Resize for inference if needed
        if max(orig_h, orig_w) > MAX_IMG_DIM:
            scale = MAX_IMG_DIM / max(orig_h, orig_w)
            img = cv2.resize(
                img,
                (int(orig_w * scale), int(orig_h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )

        resized_h, resized_w = img.shape[:2]
        scale_x = orig_w / resized_w
        scale_y = orig_h / resized_h

        # YOLO inference
        gc.collect()
        t_infer = time.time()
        try:
            with torch.no_grad():
                results = yolo(
                    img,
                    imgsz=YOLO_IMGSZ,
                    conf=YOLO_CONF,
                    iou=YOLO_IOU,
                    device="cpu",
                    verbose=False,
                )
        except Exception:
            logger.exception("[INFER] Inference crashed")
            return JSONResponse({"error": "Model inference failed."}, status_code=500)
        inf_time = time.time() - t_infer

        # Parse results — keep only military classes
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                name = class_names[cls_id] if cls_id < len(class_names) else "unknown"
                if name not in military_classes:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ox1 = int(x1 * scale_x)
                oy1 = int(y1 * scale_y)
                ox2 = int(x2 * scale_x)
                oy2 = int(y2 * scale_y)
                detections.append({
                    "class_name": name,
                    "confidence": round(conf, 4),
                    "box": [ox1, oy1, ox2, oy2],
                })
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    img,
                    f"{name} {conf:.2f}",
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

        output_path = os.path.join(OUTPUT_DIR, unique_name)
        cv2.imwrite(output_path, img)

        total_time = time.time() - req_start
        logger.info(
            "[PREDICT] infer=%.3fs total=%.3fs det=%d img=%dx%d",
            inf_time, total_time, len(detections), orig_w, orig_h,
        )

        del img, results
        gc.collect()

        return JSONResponse({
            "message": "Soldier Detected" if detections else "No Soldier Detected",
            "detections": detections,
            "image_url": f"/outputs/{unique_name}",
        })

    except Exception:
        logger.exception("[PREDICT] Unexpected error after %.2fs", time.time() - req_start)
        return JSONResponse({"error": "Internal server error."}, status_code=500)
    finally:
        if upload_path and os.path.exists(upload_path):
            os.remove(upload_path)


app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
