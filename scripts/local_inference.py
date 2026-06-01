"""
Local inference script matching production pipeline exactly.
Compares local predictions with production API predictions.
"""
import json, os, sys, time, hashlib
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import requests

# === Configuration (MUST match production) ===
YOLO_IMGSZ = 320
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
MAX_INFERENCE_SIZE = 320
CLASS_NAMES = [
    "camouflage_soldier", "weapon", "military_tank",
    "military_truck", "military_vehicle", "civilian",
    "soldier", "civilian_vehicle", "artillery",
    "military_aircraft", "warship",
]
MILITARY_CLASSES = [
    "camouflage_soldier", "weapon", "military_tank",
    "military_truck", "military_vehicle", "soldier",
    "artillery", "military_aircraft", "warship",
]

OUTPUT_DIR = "local_inference_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE = "https://military-object-detection.onrender.com"

# === Load model ===
print("Loading model...")
t0 = time.time()
model_path = hf_hub_download(
    repo_id="datasidahmed/military_object_detection",
    filename="best.pt",
)
print(f"  Model path: {model_path}")
print(f"  Model size: {os.path.getsize(model_path)} bytes")
with open(model_path, 'rb') as f:
    sha = hashlib.sha256(f.read()).hexdigest()
print(f"  SHA256: {sha}")

torch.set_num_threads(1)
model = YOLO(model_path)
print(f"  Loaded in {time.time()-t0:.1f}s")
print()

def local_predict(img_path_or_bytes):
    """
    Exact reproduction of production pipeline.
    1. cv2.imread (BGR)
    2. Resize to MAX_INFERENCE_SIZE (320)
    3. YOLO predict with imgsz=320, conf=0.25, iou=0.5
    """
    if isinstance(img_path_or_bytes, str):
        img = cv2.imread(img_path_or_bytes)
    else:
        nparr = np.frombuffer(img_path_or_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image", "detections": []}

    h, w = img.shape[:2]
    if max(h, w) > MAX_INFERENCE_SIZE:
        scale = MAX_INFERENCE_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    t0 = time.time()
    with torch.no_grad():
        results = model(
            img,
            imgsz=YOLO_IMGSZ,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device="cpu",
            verbose=False,
        )
    inf_time = time.time() - t0

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class_name": class_name,
                "confidence": round(conf, 4),
                "box": [x1, y1, x2, y2],
                "is_military": class_name in MILITARY_CLASSES,
            })

    return {"detections": detections, "inference_time_s": round(inf_time, 3), "img_shape": list(img.shape)}

def production_predict(img_bytes):
    """Send image to production API."""
    t0 = time.time()
    r = requests.post(
        f"{BASE}/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        timeout=300,
    )
    elapsed = time.time() - t0
    if r.status_code == 200:
        data = r.json()
        return {
            "detections": data.get("detections", []),
            "message": data.get("message", ""),
            "elapsed_s": round(elapsed, 2),
        }
    else:
        return {"error": f"HTTP {r.status_code}", "body": r.text[:200], "elapsed_s": round(elapsed, 2)}


def compare_detections(local, production, label=""):
    """Compare local vs production detections."""
    local_dets = local.get("detections", [])
    prod_dets = production.get("detections", [])
    
    local_military = [d for d in local_dets if d.get("is_military")]
    prod_military = [d for d in prod_dets if True]  # Production already filters
    
    print(f"\n{'='*60}")
    print(f"IMAGE: {label}")
    print(f"{'='*60}")
    print(f"  Local inference time: {local.get('inference_time_s', 'N/A')}s")
    print(f"  Production API time: {production.get('elapsed_s', 'N/A')}s")
    print(f"  Image shape (after resize): {local.get('img_shape', 'N/A')}")
    
    if local.get("error"):
        print(f"  Local ERROR: {local['error']}")
    if production.get("error"):
        print(f"  Production ERROR: {production['error']}")
        return
    
    print(f"\n  LOCAL detections ({len(local_military)} military):")
    if local_military:
        for d in local_military:
            print(f"    {d['class_name']}: conf={d['confidence']:.4f} box={d['box']}")
    else:
        print(f"    (no military detections)")
    if local_dets:
        for d in local_dets:
            if not d.get("is_military"):
                print(f"    (non-military: {d['class_name']} conf={d['confidence']:.4f})")
    
    print(f"\n  PRODUCTION detections ({len(prod_military)}):")
    if prod_military:
        for d in prod_military:
            print(f"    {d['class_name']}: conf={d['confidence']:.4f} box={d['box']}")
    else:
        print(f"    (no detections)")
    
    # Check for discrepancies
    local_names = sorted([(d['class_name'], d['confidence']) for d in local_military])
    prod_names = sorted([(d['class_name'], d['confidence']) for d in prod_military])
    
    if local_names == prod_names:
        print(f"\n  >>> MATCH: Local and production predictions identical")
    else:
        print(f"\n  >>> MISMATCH: Local and production differ!")
        print(f"  Local classes: {[n for n,c in local_names]}")
        print(f"  Prod classes:  {[n for n,c in prod_names]}")
    
    return local_names == prod_names


if __name__ == "__main__":
    print("Local inference ready. Generating test images...")
    
    # Download test images from picsum
    test_images = []
    for label, w, h in [("civilian_1", 640, 480), ("civilian_2", 800, 600), ("landscape_1", 1024, 768)]:
        r = requests.get(f"https://picsum.photos/{w}/{h}.jpg", timeout=30, allow_redirects=True)
        path = os.path.join(OUTPUT_DIR, f"{label}.jpg")
        with open(path, "wb") as f:
            f.write(r.content)
        test_images.append((label, path))
        print(f"  Downloaded {label} ({w}x{h}): {len(r.content)} bytes")
    
    # Run comparison
    matches = 0
    for label, path in test_images:
        with open(path, "rb") as f:
            img_bytes = f.read()
        local = local_predict(path)
        prod = production_predict(img_bytes)
        if compare_detections(local, prod, label):
            matches += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {matches}/{len(test_images)} images matched")
    print(f"{'='*60}")
