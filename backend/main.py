import logging
import os
import shutil
import uuid

import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ===============================
# Logging
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Military Object Detection API")

# ===============================
# CORS
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# ROUTE: Health check
# ===============================
@app.get("/health")
def health():
    return {"status": "ok"}


# ===============================
# ROUTE: الصفحة الرئيسية
# ===============================
@app.get("/")
def home():
    return FileResponse("frontend/index.html")


# ===============================
# إعداد الموديل (Lazy Loading)
# ===============================
model = None

def load_model():
    global model

    if model is None:
        logger.info("Loading model from Hugging Face...")

        cache_dir = "/tmp/hf_cache"
        os.makedirs(cache_dir, exist_ok=True)

        model_path = hf_hub_download(
            repo_id="datasidahmed/military_object_detection",
            filename="best.pt",
            token=os.getenv("HF_TOKEN"),
            cache_dir=cache_dir,
        )

        logger.info("Model downloaded: %s", model_path)
        model = YOLO(model_path)

    return model


# ===============================
# إعداد المجلدات
# ===============================
UPLOAD_DIR = "/tmp/uploads"
OUTPUT_DIR = "/tmp/outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# الكلاسات
# ===============================
military_classes = {
    "camouflage_soldier",
    "weapon",
    "military_tank",
    "military_truck",
    "military_vehicle",
    "soldier",
    "artillery",
    "military_aircraft",
    "warship",
}

class_names = [
    "camouflage_soldier", "weapon", "military_tank",
    "military_truck", "military_vehicle", "civilian",
    "soldier", "civilian_vehicle", "artillery",
    "military_aircraft", "warship",
]


# ===============================
# ROUTE: prediction
# ===============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        yolo = load_model()

        filename = os.path.basename(file.filename or "upload")

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported")

        ext = filename.rsplit(".", 1)[-1]
        unique_name = f"{uuid.uuid4()}.{ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_name)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = cv2.imread(file_path)
        if img is None:
            logger.error("cv2 failed to read image: %s", file_path)
            raise HTTPException(status_code=422, detail="Could not decode image — ensure the file is a valid JPEG or PNG")

        results = yolo(file_path, conf=0.25, iou=0.5, device="cpu")

        detections = []
        military_found = False

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])

                if cls_id >= len(class_names):
                    logger.warning("Skipping unknown class id %d", cls_id)
                    continue

                conf = float(box.conf[0])
                class_name = class_names[cls_id]
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
                        img,
                        f"{class_name} {conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

        output_path = os.path.join(OUTPUT_DIR, unique_name)
        cv2.imwrite(output_path, img)

        if not military_found:
            return JSONResponse({
                "message": "No military object detected ❌",
                "detections": [],
                "image_url": f"/outputs/{unique_name}",
            })

        return JSONResponse({
            "message": "Military objects detected ✅",
            "detections": detections,
            "image_url": f"/outputs/{unique_name}",
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error during prediction")
        raise HTTPException(status_code=500, detail=str(e))


# ===============================
# Static files (output images)
# ===============================
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
