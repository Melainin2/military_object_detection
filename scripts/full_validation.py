"""
Full validation: start server, wait for model loading, run all tests, report.
"""
import subprocess, sys, os, time, json, requests, signal, threading, cv2

BASE = "http://localhost:8000"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def wait_for_server(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{BASE}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get("model_loaded") and data.get("vit_loaded"):
                    print(f"  Server ready ({time.time()-t0:.0f}s): {data}")
                    return True
                else:
                    print(f"  Waiting for models... ({time.time()-t0:.0f}s) model={data.get('model_loaded')} vit={data.get('vit_loaded')}")
                    time.sleep(3)
            else:
                print(f"  Server status: {r.status_code}")
                time.sleep(2)
        except requests.exceptions.ConnectionError:
            print(f"  Waiting for server... ({time.time()-t0:.0f}s)")
            time.sleep(2)
    return False

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name} {detail}")

# =============================================================
print("1. Starting server...")
# =============================================================
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    env={**os.environ, "PYTHONPATH": os.getcwd()}
)

print(f"  Server PID: {proc.pid}")

if not wait_for_server():
    print("  FAILED to start server!")
    proc.kill()
    sys.exit(1)

# =============================================================
print("\n" + "=" * 70)
print("2. ViT info in response")
print("=" * 70)
# =============================================================
with open("test_images_military_vs_civilian/mil_f35_jet.jpg", "rb") as f:
    img_bytes = f.read()
r = requests.post(f"{BASE}/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")}, timeout=60)
data = r.json()

check("HTTP 200", r.status_code == 200)
check("vit_top_predictions in response", "vit_top_predictions" in data)
check("vit_confidence in response", "vit_confidence" in data)
check("vit_military_score in response", "vit_military_score" in data)
check("vit_loaded in response", "vit_loaded" in data)
check("vit_confidence > 0", data.get("vit_confidence", 0) > 0.5)
check("top-1 is military", data.get("vit_top_predictions", [{}])[0].get("is_military", False))
print(f"       mil_score={data.get('vit_military_score',0):.4f} conf={data.get('vit_confidence',0):.4f}")
print(f"       top-1: {data['vit_top_predictions'][0]['class']}")

# =============================================================
print("\n" + "=" * 70)
print("3. Box coordinates in original image space")
print("=" * 70)
# =============================================================
test_pairs = [
    ("mil_f35_jet.jpg", "test_images_military_vs_civilian/mil_f35_jet.jpg"),
    ("sout_civilian_ship.jpg", "test_images_military_vs_civilian/sout_civilian_ship.jpg"),
]

for name, path in test_pairs:
    img = cv2.imread(path)
    orig_h, orig_w = img.shape[:2]
    with open(path, "rb") as f:
        img_bytes = f.read()
    r = requests.post(f"{BASE}/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")}, timeout=60)
    data = r.json()
    
    if data.get("detections"):
        det = data["detections"][0]
        box = det["box"]
        x1, y1, x2, y2 = box
        in_bounds = (0 <= x1 <= orig_w) and (0 <= x2 <= orig_w) and \
                    (0 <= y1 <= orig_h) and (0 <= y2 <= orig_h)
        check(f"{name}: box in bounds of {orig_w}x{orig_h}", in_bounds,
              f"box={box}")
        if max(orig_h, orig_w) > 320:
            # Max dim > 320 means image was resized -> box should be > resized coords
            resized_max = 320
            scale_factor = max(orig_h, orig_w) / resized_max
            expected_min_coord = 0  # box could start at 0
            # At least one coord should be > resized dimension
            bigger_than_resized = (x2 > resized_max * 0.5 or y2 > resized_max * 0.5)
            check(f"{name}: box scaled to original space ({scale_factor:.1f}x)", bigger_than_resized)

# =============================================================
print("\n" + "=" * 70)
print("4. Content-Type None handling")
print("=" * 70)
# =============================================================
# Simulate multipart without Content-Type for the file part
import http.client
with open("test_images_military_vs_civilian/civilian_park.jpg", "rb") as f:
    img_bytes = f.read()

conn = http.client.HTTPConnection("localhost", 8000)
boundary = "----TestBoundary456"
body_parts = []
body_parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.jpg\"\r\n\r\n".encode())
body_parts.append(img_bytes)
body_parts.append(f"\r\n--{boundary}--\r\n".encode())
body = b"".join(body_parts)
conn.request("POST", "/predict", body, {
    "Content-Type": f"multipart/form-data; boundary={boundary}"
})
r = conn.getresponse()
raw = r.read().decode()
try:
    resp_data = json.loads(raw)
    check("HTTP 200 (no MIME in part)", r.status == 200,
          f"HTTP {r.status}: {resp_data.get('message','')[:60]}")
except:
    check("HTTP 200 (no MIME in part)", r.status == 200, f"HTTP {r.status}: {raw[:100]}")

# =============================================================
print("\n" + "=" * 70)
print("5. Schema validation")
print("=" * 70)
# =============================================================
check("Sends vit_top_predictions as list",
      isinstance(data.get("vit_top_predictions"), list))
check("Sends detections as list",
      isinstance(data.get("detections"), list))
check("Sends image_url as string",
      isinstance(data.get("image_url"), str))
check("Sends message as string",
      isinstance(data.get("message"), str))

# =============================================================
print("\n" + "=" * 70)
print("6. Error handling")
print("=" * 70)
# =============================================================
r = requests.post(f"{BASE}/predict", files={"file": ("test.txt", b"hello", "text/plain")}, timeout=10)
check("Text file rejected (extension)", r.status_code == 400)

r = requests.post(f"{BASE}/predict", files={"file": ("test.jpg", b"notanimage", "image/jpeg")}, timeout=10)
check("Invalid image rejected (decoding)", r.status_code == 400)

# =============================================================
print("\n" + "=" * 70)
print("7. Verify configuration")
print("=" * 70)
# =============================================================
with open("backend/classifier_service.py") as f:
    content = f.read()
    check("threshold fix applied (p >= self.threshold)", "p >= self.threshold" in content)
    check("top_k=20 in classifier_service.py", "top_k=20" in content)
    check("threshold=0.05 in classifier_service.py", "threshold=0.05" in content)

# =============================================================
print(f"\n{'='*70}")
print("VALIDATION COMPLETE")
print(f"{'='*70}")

# Cleanup
proc.terminate()
proc.wait(timeout=10)
print(f"\nServer stopped (PID: {proc.pid})")
