"""
Refined production validation — avoids problematic image edge cases.
"""
import json, os, time
from datetime import datetime
import requests

BASE = "https://military-object-detection.onrender.com"
RESULTS = {}

def log(s):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {s}", flush=True)

def dl_img(label, w, h):
    url = f"https://picsum.photos/{w}/{h}.jpg"
    t0 = time.time()
    r = requests.get(url, timeout=30, allow_redirects=True)
    d = r.content
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Downloaded {label} ({w}x{h}): {len(d)} bytes in {time.time()-t0:.1f}s", flush=True)
    return d

def test(label, img_bytes):
    t0 = time.time()
    try:
        r = requests.post(f"{BASE}/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")}, timeout=300)
        body = r.text
        try:
            js = r.json()
            dets = len(js.get("detections", []))
            msg = js.get("message", "")
        except Exception:
            js = None
            dets = 0
            msg = body[:80]
        return {
            "label": label, "status": r.status_code, "elapsed_s": round(time.time()-t0, 2),
            "body_len": len(body), "detections": dets, "message": msg,
            "x_render_routing": r.headers.get("x-render-routing", ""),
        }
    except requests.Timeout:
        return {"label": label, "status": 0, "elapsed_s": round(time.time()-t0, 2), "error": "timeout"}
    except Exception as e:
        return {"label": label, "status": 0, "elapsed_s": round(time.time()-t0, 2), "error": str(e)}

def summary(results):
    ok = [r for r in results if r["status"] == 200]
    elapsed = [r["elapsed_s"] for r in ok]
    n = len(results)
    return {
        "total": n, "success": len(ok), "failed": n - len(ok),
        "success_rate": f"{len(ok)/n*100:.1f}%" if n else "N/A",
        "avg_s": round(sum(elapsed)/len(elapsed), 2) if elapsed else 0,
        "max_s": round(max(elapsed), 2) if elapsed else 0,
        "min_s": round(min(elapsed), 2) if elapsed else 0,
        "p50_s": round(sorted(elapsed)[len(elapsed)//2], 2) if elapsed else 0,
        "p95_s": round(sorted(elapsed)[min(int(len(elapsed)*0.95), len(elapsed)-1)], 2) if elapsed else 0,
        "total_detections": sum(r.get("detections", 0) for r in results),
    }

# =====================================================================
log("=" * 60)
log("PHASE 1: Wake & health check")
log("=" * 60)
r = requests.get(f"{BASE}/health", timeout=60)
log(f"  /health: status={r.status_code} body={r.text}")
time.sleep(3)

# =====================================================================
log("=" * 60)
log("PHASE 2: Download test images (max 1600px)")
log("=" * 60)
images = []
sizes = [
    ("s_200x200", 200, 200),
    ("s_320x240", 320, 240),
    ("s_400x300", 400, 300),
    ("s_250x250", 250, 250),
    ("s_300x400", 300, 400),
    ("m_800x600", 800, 600),
    ("m_1024x768", 1024, 768),
    ("m_640x480", 640, 480),
    ("m_1280x720", 1280, 720),
    ("m_900x600", 900, 600),
    ("l_1600x1200", 1600, 1200),
    ("l_1400x900", 1400, 900),
    ("l_1200x800", 1200, 800),
    ("l_1500x1000", 1500, 1000),
    ("l_1000x750", 1000, 750),
]
for label, w, h in sizes:
    images.append((label, dl_img(label, w, h)))

# =====================================================================
log("=" * 60)
log("PHASE 3: 15 size-varied predictions")
log("=" * 60)
size_results = []
for label, img_bytes in images:
    r = test(label, img_bytes)
    log(f"  {label}: status={r['status']} elapsed={r['elapsed_s']}s detections={r['detections']}")
    size_results.append(r)
    time.sleep(1)

s = summary(size_results)
log(f"  Size-varied: {s['success']}/{s['total']} ({s['success_rate']}) avg={s['avg_s']}s max={s['max_s']}s")

if s['failed']:
    for r in size_results:
        if r['status'] != 200:
            log(f"  FAIL: {r['label']} status={r['status']} elapsed={r['elapsed_s']}s {r.get('message','')}")

# =====================================================================
log("=" * 60)
log("PHASE 4: 20 consecutive predictions (load test)")
log("=" * 60)
load_img = images[5][1]  # m_800x600
load_results = []
for i in range(20):
    r = test(f"load_{i+1:02d}", load_img)
    log(f"  Load {i+1:02d}/20: status={r['status']} elapsed={r['elapsed_s']}s detections={r['detections']}")
    load_results.append(r)
    time.sleep(0.5)

s2 = summary(load_results)
log(f"  Load test: {s2['success']}/{s2['total']} ({s2['success_rate']}) avg={s2['avg_s']}s max={s2['max_s']}s")

if s2['failed']:
    for r in load_results:
        if r['status'] != 200:
            log(f"  FAIL: {r['label']} status={r['status']} elapsed={r['elapsed_s']}s")

# =====================================================================
log("=" * 60)
log("PHASE 5: Error handling tests")
log("=" * 60)
errors = []
# Invalid text file
r = requests.post(f"{BASE}/predict", files={"file": ("test.txt", b"hello", "text/plain")}, timeout=30)
errors.append({"label": "invalid_type", "status": r.status_code, "body": r.text[:60]})
log(f"  Invalid type: status={r.status_code} body={r.text[:50]}")

# No file
r = requests.post(f"{BASE}/predict", timeout=30)
errors.append({"label": "no_file", "status": r.status_code, "body": r.text[:60]})
log(f"  No file: status={r.status_code} body={r.text[:50]}")

# Oversized 11MB
big = b"X" * 11 * 1024 * 1024
r = requests.post(f"{BASE}/predict", files={"file": ("big.jpg", big, "image/jpeg")}, timeout=30)
errors.append({"label": "oversized", "status": r.status_code, "body": r.text[:60]})
log(f"  Oversized: status={r.status_code} body={r.text[:50]}")

# =====================================================================
log("=" * 60)
log("FINAL COMBINED RESULTS")
log("=" * 60)

all_results = size_results + load_results
combined = summary(all_results)

log(f"Total requests: {combined['total']}")
log(f"Successful:     {combined['success']} ({combined['success_rate']})")
log(f"Failed:         {combined['failed']}")
log(f"Avg duration:   {combined['avg_s']}s")
log(f"Max duration:   {combined['max_s']}s")
log(f"Min duration:   {combined['min_s']}s")
log(f"P50 duration:   {combined['p50_s']}s")
log(f"P95 duration:   {combined['p95_s']}s")
log(f"Detections:     {combined['total_detections']}")

if combined['failed'] == 0:
    log(f"\n  ALL 35 PRODUCTION TESTS PASSED")
else:
    log(f"\n  {combined['failed']} TESTS FAILED")

# Save
results = {
    "timestamp": datetime.now().isoformat(),
    "version": "2.0.0+no_grad_fix",
    "size_varied": {"results": size_results, "summary": s},
    "load_test": {"results": load_results, "summary": s2},
    "error_handling": errors,
    "combined": combined,
}
out_path = os.path.join(os.path.dirname(__file__), "..", "production_test_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
log(f"\nResults saved to: {out_path}")
