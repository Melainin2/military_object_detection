import logging
import os
import uuid
import gc
import time
import threading

import numpy as np
import cv2
import onnxruntime as ort

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download

try:
    from classifier_service import ImageClassifier  # Render/Docker (backend/ root)
except ModuleNotFoundError:
    from backend.classifier_service import ImageClassifier  # local dev

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backend")

app = FastAPI(title="AI Detection Dashboard", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR  = "uploads"
OUTPUT_DIR  = "outputs"
YOLO_IMGSZ  = 320
YOLO_CONF   = 0.35
YOLO_IOU    = 0.45
MAX_INPUT_DIM = 1280   # cap large uploads before letterbox
MAX_BOX_AREA_RATIO = 0.70  # ignore boxes covering >70% of image (portrait false-positives)
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

HF_REPO_ID    = "datasidahmed/YOLOV8"
HF_FILENAME   = "best.onnx"
MODEL_CACHE   = "best.onnx"

session          = None
model_loaded     = False
model_lock       = threading.Lock()
classifier       = None
vit_loaded       = False
vit_lock         = threading.Lock()

CLASS_NAMES = [
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

MILITARY_CLASSES = {
    "camouflage_soldier", "weapon", "military_tank", "military_truck",
    "military_vehicle", "soldier", "military_artillery", "trench",
    "military_aircraft", "military_warship",
}


# ── ONNX preprocessing ───────────────────────────────────────────────────────

def _letterbox(img, size=320, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h = (size - nh) / 2
    pad_w = (size - nw) / 2
    top  = int(round(pad_h - 0.1))
    bot  = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right= int(round(pad_w + 0.1))
    img = cv2.copyMakeBorder(img, top, bot, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, left, top


def _preprocess(img_bgr):
    h, w = img_bgr.shape[:2]
    if max(h, w) > MAX_INPUT_DIM:
        scale = MAX_INPUT_DIM / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    lb, ratio, pad_w, pad_h = _letterbox(img_bgr, YOLO_IMGSZ)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = rgb.transpose(2, 0, 1)[np.newaxis]   # NCHW
    return tensor, ratio, pad_w, pad_h


def _postprocess(raw, ratio, pad_w, pad_h):
    """
    raw: [1, 16, 2100] — cx cy w h + 12 class scores (pixel coords in letterbox)
    Returns list of {class_id, confidence, box [x1,y1,x2,y2] in original coords}
    """
    pred = raw[0].T                 # [2100, 16]
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    scores = pred[:, 4:]            # [2100, 12]

    class_ids   = np.argmax(scores, axis=1)
    confidences = scores[np.arange(len(scores)), class_ids]

    mask = confidences >= YOLO_CONF
    if not mask.any():
        return []

    cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
    confidences  = confidences[mask]
    class_ids    = class_ids[mask]

    # xywh → xyxy in letterbox coords
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # Undo letterbox padding and scale → original image coords
    x1 = (x1 - pad_w) / ratio
    y1 = (y1 - pad_h) / ratio
    x2 = (x2 - pad_w) / ratio
    y2 = (y2 - pad_h) / ratio

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # OpenCV NMS (expects xywh)
    boxes_xywh = [[float(x1[i]), float(y1[i]), float(x2[i]-x1[i]), float(y2[i]-y1[i])]
                  for i in range(len(x1))]
    idxs = cv2.dnn.NMSBoxes(boxes_xywh, confidences.tolist(), YOLO_CONF, YOLO_IOU)

    results = []
    for i in (idxs.flatten() if len(idxs) else []):
        results.append({
            "class_id":   int(class_ids[i]),
            "confidence": float(confidences[i]),
            "box": [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
        })
    return results


# ── Model loading ─────────────────────────────────────────────────────────────

def load_session() -> ort.InferenceSession:
    global session, model_loaded

    if session is not None:
        return session

    with model_lock:
        if session is not None:
            return session

        if os.path.exists(MODEL_CACHE):
            logger.info("[MODEL] Using cached ONNX: %s", MODEL_CACHE)
            model_path = MODEL_CACHE
        else:
            logger.info("[MODEL] Downloading ONNX from HuggingFace: %s/%s",
                        HF_REPO_ID, HF_FILENAME)
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
                raise RuntimeError(
                    f"HuggingFace download failed ({HF_REPO_ID}): {exc}"
                ) from exc

        logger.info("[MODEL] Loading ONNX session...")
        t1 = time.time()
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_mem_pattern = True
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        model_loaded = True
        logger.info("[MODEL] ONNX session ready in %.2fs", time.time() - t1)
        return session


def _warmup():
    dummy = np.zeros((1, 3, YOLO_IMGSZ, YOLO_IMGSZ), dtype=np.float32)
    sess = load_session()
    inp_name = sess.get_inputs()[0].name
    sess.run(None, {inp_name: dummy})
    logger.info("[MODEL] Warmup complete")


def _startup():
    try:
        _warmup()
    except Exception:
        logger.exception("[MODEL] Startup failed — first request will retry")


def load_vit():
    global classifier, vit_loaded
    with vit_lock:
        if classifier is not None:
            return
        try:
            logger.info("[VIT] Initializing ImageClassifier (ResNet50)...")
            t0 = time.time()
            classifier = ImageClassifier()
            vit_loaded = classifier.model_loaded
            logger.info("[VIT] Ready in %.2fs (loaded=%s)", time.time() - t0, vit_loaded)
        except Exception as exc:
            logger.exception("[VIT] Initialization failed: %s", exc)
            vit_loaded = False


@app.on_event("startup")
def on_startup():
    logger.info("=== AI Detection Dashboard v4 (ONNX) starting ===")
    threading.Thread(target=_startup, daemon=True).start()
    threading.Thread(target=load_vit, daemon=True).start()


@app.on_event("shutdown")
def on_shutdown():
    cv2.destroyAllWindows()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "vit_loaded": vit_loaded,
        "model_source": f"huggingface.co/{HF_REPO_ID}",
        "runtime": "onnxruntime",
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
                {"error": "Seuls les fichiers image JPG, PNG, WEBP sont acceptés."},
                status_code=400,
            )
        if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
            return JSONResponse(
                {"error": "Seuls les fichiers image JPG, PNG, WEBP sont acceptés."},
                status_code=400,
            )

        # Load ONNX session (cached)
        try:
            sess = load_session()
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)

        inp_name  = sess.get_inputs()[0].name
        out_name  = sess.get_outputs()[0].name

        # Save upload
        unique_name = f"{uuid.uuid4()}{ext}"
        upload_path = os.path.join(UPLOAD_DIR, unique_name)
        with open(upload_path, "wb") as buf:
            while chunk := await file.read(8192):
                buf.write(chunk)

        # Decode
        img_bgr = cv2.imread(upload_path)
        if img_bgr is None:
            return JSONResponse({"error": "Image invalide ou illisible."}, status_code=400)

        orig_h, orig_w = img_bgr.shape[:2]

        # ── STEP 1: ViT classifier ────────────────────────────────────────────
        vit_result = None
        vit_approved = False
        if classifier is not None and classifier.model_loaded:
            try:
                vit_result = classifier.predict(img_bgr)
                vit_approved = vit_result.get("is_military", False)
            except Exception:
                logger.exception("[VIT] Prediction error — falling back to YOLO-only")

        vit_score = round(vit_result["military_score"], 4) if vit_result else 0.0
        vit_conf  = round(vit_result["confidence"], 4)    if vit_result else 0.0
        vit_top   = vit_result["top_predictions"]          if vit_result else []

        if vit_result is not None:
            logger.info(
                "[DECISION] is_military=%s vit_score=%.4f vit_conf=%.4f vit_approved=%s",
                vit_result["is_military"], vit_score, vit_conf, vit_approved,
            )

        # ── STEP 2: YOLO inference ────────────────────────────────────────────
        tensor, ratio, pad_w, pad_h = _preprocess(img_bgr)

        t_infer = time.time()
        try:
            raw = sess.run([out_name], {inp_name: tensor})
        except Exception:
            logger.exception("[INFER] ONNX inference error")
            return JSONResponse({"error": "Inference failed."}, status_code=500)
        inf_ms = (time.time() - t_infer) * 1000

        boxes = _postprocess(raw[0], ratio, pad_w, pad_h)

        # Filter military only + draw
        img_area = orig_w * orig_h
        detections = []
        img_draw = img_bgr.copy()
        for b in boxes:
            name = CLASS_NAMES[b["class_id"]] if b["class_id"] < len(CLASS_NAMES) else "unknown"
            if name not in MILITARY_CLASSES:
                continue
            x1, y1, x2, y2 = b["box"]
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))
            box_area = (x2 - x1) * (y2 - y1)
            if img_area > 0 and box_area / img_area > MAX_BOX_AREA_RATIO:
                logger.info("[FILTER] Skipped large-box false-positive (%.0f%% of image)", 100 * box_area / img_area)
                continue
            conf = b["confidence"]
            detections.append({
                "class_name": name,
                "confidence": round(conf, 4),
                "box": [x1, y1, x2, y2],
            })
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img_draw, f"{name} {conf:.2f}",
                        (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        output_path = os.path.join(OUTPUT_DIR, unique_name)
        cv2.imwrite(output_path, img_draw)

        yolo_found = len(detections) > 0

        # ── FINAL DECISION ────────────────────────────────────────────────────
        # Rule A: ViT + YOLO (both agree) → Soldier Detected
        # Rule B: YOLO only → Soldier Detected
        # Rule C: ViT only → Soldier Detected (ViT trumps YOLO)
        # Rule D: Neither → No Soldier Detected
        military_found = vit_approved or yolo_found
        message = "Soldier Detected" if military_found else "No Soldier Detected"

        total_ms = (time.time() - req_start) * 1000
        logger.info(
            "[FINAL_DECISION] vit_approved=%s yolo_found=%s military_found=%s message=%s infer=%.0fms total=%.0fms",
            vit_approved, yolo_found, military_found, message, inf_ms, total_ms,
        )

        del img_bgr, img_draw, tensor, raw
        gc.collect()

        return JSONResponse({
            "message": message,
            "detections": detections,
            "image_url": f"/outputs/{unique_name}",
            "vit_loaded": classifier is not None and classifier.model_loaded,
            "vit_military_score": vit_score,
            "vit_confidence": vit_conf,
            "vit_top_predictions": vit_top,
        })

    except Exception:
        logger.exception("[PREDICT] Unexpected error after %.0fms",
                         (time.time() - req_start) * 1000)
        return JSONResponse({"error": "Internal server error."}, status_code=500)
    finally:
        if upload_path and os.path.exists(upload_path):
            os.remove(upload_path)


app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
