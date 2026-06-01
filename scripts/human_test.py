"""Test local vs production on human images to reproduce false positives."""
import cv2, torch, numpy as np, requests, time, os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

YOLO_IMGSZ = 320
CONF = 0.25
IOU = 0.5
MAX_SZ = 320
CLASS_NAMES = [
    'camouflage_soldier', 'weapon', 'military_tank', 'military_truck',
    'military_vehicle', 'civilian', 'soldier', 'civilian_vehicle',
    'artillery', 'military_aircraft', 'warship',
]
MIL_CLASSES = set([
    'camouflage_soldier', 'weapon', 'military_tank', 'military_truck',
    'military_vehicle', 'soldier', 'artillery', 'military_aircraft', 'warship',
])

torch.set_num_threads(1)
model_path = hf_hub_download(repo_id='datasidahmed/military_object_detection', filename='best.pt')
model = YOLO(model_path)
print('Model loaded')

BASE = 'https://military-object-detection.onrender.com'

def local_predict(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return []
    h, w = img.shape[:2]
    if max(h, w) > MAX_SZ:
        s = MAX_SZ / max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_LINEAR)
    with torch.no_grad():
        results = model(img, imgsz=YOLO_IMGSZ, conf=CONF, iou=IOU, device='cpu', verbose=False)
    dets = []
    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cn = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else 'unknown'
            dets.append({'class': cn, 'conf': float(box.conf[0]),
                        'box': list(map(int, box.xyxy[0])), 'military': cn in MIL_CLASSES})
    return dets

def prod_predict(img_bytes):
    t0 = time.time()
    r = requests.post(f'{BASE}/predict', files={'file': ('t.jpg', img_bytes, 'image/jpeg')}, timeout=300)
    if r.status_code == 200:
        return r.json().get('detections', []), time.time()-t0
    return [], time.time()-t0

test_urls = [
    ('person_outdoor', 'https://picsum.photos/seed/p1/640/480'),
    ('person_portrait', 'https://picsum.photos/seed/pr1/400/500'),
    ('group_people', 'https://picsum.photos/seed/gr1/800/600'),
    ('person_distant', 'https://picsum.photos/seed/d1/1024/768'),
    ('person_walking', 'https://picsum.photos/seed/w1/640/480'),
    ('crowd', 'https://picsum.photos/seed/c1/800/600'),
]

for label, url in test_urls:
    r = requests.get(url, timeout=30, allow_redirects=True)
    img = r.content
    print(f'\n=== {label} ({len(img)} bytes) ===')

    local = local_predict(img)
    prod, ptime = prod_predict(img)

    local_mil = [d for d in local if d['military']]
    prod_mil = prod

    print(f'  Local ({len(local)} total, {len(local_mil)} military):')
    for d in local_mil:
        print(f'    MIL: {d["class"]} conf={d["conf"]:.4f} box={d["box"]}')
    for d in local:
        if not d['military']:
            print(f'    CIV: {d["class"]} conf={d["conf"]:.4f}')
    if not local:
        print('    (no detections)')

    print(f'  Production ({len(prod)} detections, {ptime:.0f}s):')
    for d in prod_mil:
        print(f'    {d["class_name"]} conf={d["confidence"]:.4f} box={d["box"]}')
    if not prod_mil:
        print('    (no detections)')

    local_key = sorted([(d['class'], round(d['conf'],4)) for d in local_mil])
    prod_key = sorted([(d['class_name'], d['confidence']) for d in prod_mil])
    if local_key == prod_key:
        print('  MATCH')
    else:
        print('  MISMATCH!')
        print(f'  Local keys: {local_key}')
        print(f'  Prod keys:  {prod_key}')
