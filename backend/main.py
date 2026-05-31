import logging
import shutil
import os
import uuid

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2

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
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("Upload and output directories ready")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN is not set — model download may fail for gated models")
    else:
        logger.info("HF_TOKEN is configured")

    logger.info("Startup complete — ready for requests")


@app.on_event("shutdown")
def on_shutdown():
    logger.info("Shutting down AI Detection Dashboard")


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
    try:
        client_ip = request.client.host if request and request.client else "unknown"
        filename = os.path.basename(file.filename or "")
        logger.info("Predict request from %s — file: %s", client_ip, filename)

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            return JSONResponse({"error": "Only JPG/PNG images are allowed"}, status_code=400)

        ext = filename.rsplit(".", 1)[-1]
        unique_name = f"{uuid.uuid4()}.{ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_name)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = cv2.imread(file_path)
        if img is None:
            logger.error("cv2.imread failed for %s", file_path)
            return JSONResponse({"error": "Invalid image file"}, status_code=400)

        loaded_model = load_model()
        results = loaded_model(file_path, conf=0.25, iou=0.5, device="cpu")

        detections = []
        military_found = False

        for r in results:
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

        logger.info(
            "Processed %s from %s — %d military detections",
            unique_name, client_ip, len(detections),
        )

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
        logger.exception("Prediction failed")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


# ===============================
# Static Files
# ===============================
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")