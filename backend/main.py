from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO
from huggingface_hub import hf_hub_download

import shutil, os, cv2, uuid

app = FastAPI()

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
        print("🔄 Loading model from Hugging Face...")

        model_path = hf_hub_download(
            repo_id="datasidahmed/military_object_detection",
            filename="best.pt",
            token=os.getenv("HF_TOKEN")  # مهم
        )

        print("✅ Model downloaded:", model_path)

        model = YOLO(model_path)

    return model


# ===============================
# إعداد المجلدات
# ===============================
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# الكلاسات
# ===============================
military_classes = [
    "camouflage_soldier",
    "weapon",
    "military_tank",
    "military_truck",
    "military_vehicle",
    "soldier",
    "artillery",
    "military_aircraft",
    "warship"
]

class_names = [
    "camouflage_soldier","weapon","military_tank",
    "military_truck","military_vehicle","civilian",
    "soldier","civilian_vehicle","artillery",
    "military_aircraft","warship"
]


# ===============================
# ROUTE: prediction
# ===============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        model = load_model()

        filename = os.path.basename(file.filename)

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            return JSONResponse({"error": "Only images allowed"})

        # اسم ملف unique
        ext = filename.split(".")[-1]
        unique_name = f"{uuid.uuid4()}.{ext}"

        file_path = os.path.join(UPLOAD_DIR, unique_name)

        # حفظ الصورة
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # قراءة الصورة
        img = cv2.imread(file_path)

        if img is None:
            print("❌ cv2 failed to read image")
            return JSONResponse({"error": "Invalid image"})

        # تشغيل الموديل
        results = model(file_path, conf=0.25, iou=0.5, device="cpu")

        detections = []
        military_found = False

        for r in results:
            for box in r.boxes:

                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = class_names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if class_name in military_classes:
                    military_found = True

                    detections.append({
                        "class_name": class_name,
                        "confidence": conf,
                        "box": [x1, y1, x2, y2]
                    })

                    # رسم
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.putText(img,
                                f"{class_name} {conf:.2f}",
                                (x1,y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0,0,255),
                                2)

        # حفظ الصورة الناتجة
        output_path = os.path.join(OUTPUT_DIR, unique_name)
        cv2.imwrite(output_path, img)

        if not military_found:
            return JSONResponse({
                "message": "No military object detected ❌",
                "detections": [],
                "image_url": f"/outputs/{unique_name}"
            })

        return JSONResponse({
            "message": "Military objects detected ✅",
            "detections": detections,
            "image_url": f"/outputs/{unique_name}"
        })

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return JSONResponse({"error": str(e)}, status_code=500)


# ===============================
# Static files (output images)
# ===============================
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")