"""
Production validation test for military-object-detection.onrender.com
Tests predict endpoint with varied image sizes, records detailed metrics.
"""
import json, os, sys, time
from datetime import datetime
import requests

BASE = "https://military-object-detection.onrender.com"
RESULTS = {}

def log(s):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {s}", flush=True)

def dl_img(label, w, h):
    """Download a random image from picsum.photos."""
    url = f"https://picsum.photos/{w}/{h}.jpg"
    t0 = time.time()
    r = requests.get(url, timeout=30, allow_redirects=True)
    elapsed = time.time() - t0
    data = r.content
    log(f"  Downloaded {label} ({w}x{h}): {len(data)} bytes in {elapsed:.1f}s")
    return data, elapsed

def test_predict(label, img_bytes):
    """Send one predict request, measure everything."""
    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE}/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            timeout=300,
        )
        elapsed = time.time() - t0
        body = r.text
        try:
            js = r.json()
        except Exception:
            js = None
        return {
            "label": label,
            "status": r.status_code,
            "elapsed_s": round(elapsed, 2),
            "body_len": len(body),
            "json": js is not None,
            "detections": len(js.get("detections", [])) if js else 0,
            "message": js.get("message", "") if js else body[:80],
            "x_render_routing": r.headers.get("x-render-routing", ""),
            "date": r.headers.get("date", ""),
            "error": None,
        }
    except requests.Timeout:
        elapsed = time.time() - t0
        return {
            "label": label,
            "status": 0,
            "elapsed_s": round(elapsed, 2),
            "body_len": 0,
            "json": False,
            "detections": 0,
            "message": "TIMEOUT",
            "x_render_routing": "",
            "date": "",
            "error": "timeout",
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "label": label,
            "status": 0,
            "elapsed_s": round(elapsed, 2),
            "body_len": 0,
            "json": False,
            "detections": 0,
            "message": str(e)[:80],
            "x_render_routing": "",
            "date": "",
            "error": str(e),
        }

def analyze(results):
    """Compute summary stats from results list."""
    statuses = [r["status"] for r in results]
    elapsed = [r["elapsed_s"] for r in results if r["status"] != 0]
    dets = [r["detections"] for r in results]
    errors = [r for r in results if r["error"] or r["status"] not in (200,)]
    ok = [r for r in results if r["status"] == 200]
    n = len(results)
    return {
        "total": n,
        "success": len(ok),
        "failed": n - len(ok),
        "errors": len(errors),
        "success_rate_pct": round(len(ok) / n * 100, 1) if n else 0,
        "status_200": sum(1 for s in statuses if s == 200),
        "status_502": sum(1 for s in statuses if s == 502),
        "status_0": sum(1 for s in statuses if s == 0),
        "avg_elapsed_s": round(sum(elapsed) / len(elapsed), 2) if elapsed else 0,
        "max_elapsed_s": round(max(elapsed), 2) if elapsed else 0,
        "min_elapsed_s": round(min(elapsed), 2) if elapsed else 0,
        "p50_elapsed_s": round(sorted(elapsed)[len(elapsed)//2], 2) if elapsed else 0,
        "p95_elapsed_s": round(sorted(elapsed)[int(len(elapsed)*0.95)], 2) if len(elapsed) >= 20 else round(sorted(elapsed)[-1], 2) if elapsed else 0,
        "total_detections": sum(dets),
        "failed_details": [
            {"label": r["label"], "status": r["status"], "elapsed_s": r["elapsed_s"], "error": r["error"]}
            for r in results if r["error"] or r["status"] not in (200,)
        ],
    }


# =====================================================================
# Phase 1: Warm-up / Wake-up
# =====================================================================
log("=" * 60)
log("PHASE 1: Wake service")
log("=" * 60)

t0 = time.time()
try:
    r = requests.get(f"{BASE}/health", timeout=60)
    log(f"  /health status={r.status_code} body={r.text}")
    RESULTS["wake"] = {"status": r.status_code, "body": r.text, "elapsed_s": round(time.time() - t0, 2)}
except Exception as e:
    log(f"  /health failed: {e}")
    RESULTS["wake"] = {"error": str(e)}

# Wait a few seconds for model to load if needed
time.sleep(3)

# =====================================================================
# Phase 2: Download test images
# =====================================================================
log("=" * 60)
log("PHASE 2: Download test images")
log("=" * 60)

images = []
sizes = [
    ("small_01", 200, 200),
    ("small_02", 320, 240),
    ("small_03", 400, 300),
    ("small_04", 250, 250),
    ("small_05", 300, 400),
    ("medium_01", 800, 600),
    ("medium_02", 1024, 768),
    ("medium_03", 640, 480),
    ("medium_04", 1280, 720),
    ("medium_05", 900, 600),
    ("large_01", 1920, 1080),
    ("large_02", 2560, 1440),
    ("large_03", 3840, 2160),
    ("large_04", 2000, 1500),
    ("large_05", 1600, 1200),
]

for label, w, h in sizes:
    data, dl_t = dl_img(label, w, h)
    images.append((label, data))

# =====================================================================
# Phase 3: Size-varied predictions (5 small, 5 medium, 5 large)
# =====================================================================
log("=" * 60)
log("PHASE 3: Size-varied predictions (15 requests)")
log("=" * 60)

size_results = []
for label, img_bytes in images:
    log(f"  Testing {label} ({len(img_bytes)} bytes)...")
    result = test_predict(label, img_bytes)
    log(f"    -> status={result['status']} elapsed={result['elapsed_s']}s detections={result['detections']}")
    size_results.append(result)
    # Small delay between requests to not overwhelm
    time.sleep(1)

RESULTS["size_varied"] = {
    "results": size_results,
    "summary": analyze(size_results),
}

log(f"\n  Size-varied summary:")
s = RESULTS["size_varied"]["summary"]
log(f"    Success: {s['success']}/{s['total']} ({s['success_rate_pct']}%)")
log(f"    Avg: {s['avg_elapsed_s']}s  Max: {s['max_elapsed_s']}s  P95: {s['p95_elapsed_s']}s")
if s["failed_details"]:
    for f in s["failed_details"]:
        log(f"    FAILED: {f['label']} status={f['status']} elapsed={f['elapsed_s']}s error={f['error']}")

# =====================================================================
# Phase 4: Load test (20 consecutive predictions)
# =====================================================================
log("=" * 60)
log("PHASE 4: Load test (20 consecutive predictions)")
log("=" * 60)

# Use a medium image
load_img = [d for l, d in images if l == "medium_03"][0]  # 640x480

load_results = []
for i in range(20):
    label = f"load_{i+1:02d}"
    log(f"  Load test {i+1}/20...")
    result = test_predict(label, load_img)
    log(f"    -> status={result['status']} elapsed={result['elapsed_s']}s detections={result['detections']}")
    load_results.append(result)
    time.sleep(0.5)

RESULTS["load_test"] = {
    "results": load_results,
    "summary": analyze(load_results),
}

log(f"\n  Load test summary:")
s = RESULTS["load_test"]["summary"]
log(f"    Success: {s['success']}/{s['total']} ({s['success_rate_pct']}%)")
log(f"    Avg: {s['avg_elapsed_s']}s  Max: {s['max_elapsed_s']}s  P95: {s['p95_elapsed_s']}s")
if s["failed_details"]:
    for f in s["failed_details"]:
        log(f"    FAILED: {f['label']} status={f['status']} elapsed={f['elapsed_s']}s error={f['error']}")

# =====================================================================
# Phase 5: Error handling tests
# =====================================================================
log("=" * 60)
log("PHASE 5: Error handling tests")
log("=" * 60)

error_tests = []

# Invalid file (not an image)
r = requests.post(f"{BASE}/predict", files={"file": ("test.txt", b"hello world", "text/plain")}, timeout=30)
error_tests.append({
    "label": "invalid_file",
    "status": r.status_code,
    "body": r.text[:100],
    "elapsed_s": round(r.elapsed.total_seconds(), 2),
})
log(f"  Invalid file: status={r.status_code} body={r.text[:60]}")

# No file
try:
    r = requests.post(f"{BASE}/predict", timeout=30)
    error_tests.append({
        "label": "no_file",
        "status": r.status_code,
        "body": r.text[:100],
        "elapsed_s": round(r.elapsed.total_seconds(), 2),
    })
    log(f"  No file: status={r.status_code} body={r.text[:60]}")
except Exception as e:
    error_tests.append({"label": "no_file", "error": str(e)})
    log(f"  No file: ERROR {e}")

# Large file (should be rejected)
big_data = b"0" * 15 * 1024 * 1024  # 15 MB
r = requests.post(f"{BASE}/predict", files={"file": ("big.jpg", big_data, "image/jpeg")}, timeout=30)
error_tests.append({
    "label": "oversized",
    "status": r.status_code,
    "body": r.text[:100],
    "elapsed_s": round(r.elapsed.total_seconds(), 2),
})
log(f"  Oversized: status={r.status_code} body={r.text[:60]}")

RESULTS["error_handling"] = error_tests

# =====================================================================
# Phase 6: Combined analysis
# =====================================================================
log("=" * 60)
log("PHASE 6: Combined analysis")
log("=" * 60)

all_results = size_results + load_results
combined = analyze(all_results)
RESULTS["combined"] = combined

log(f"  Total requests: {combined['total']}")
log(f"  Success: {combined['success']} ({combined['success_rate_pct']}%)")
log(f"  Failed: {combined['failed']}")
log(f"  Avg duration: {combined['avg_elapsed_s']}s")
log(f"  Max duration: {combined['max_elapsed_s']}s")
log(f"  P95 duration: {combined['p95_elapsed_s']}s")
log(f"  Status 200: {combined['status_200']}")
log(f"  Status 502: {combined['status_502']}")
log(f"  Total detections: {combined['total_detections']}")

if combined["failed_details"]:
    log(f"\n  FAILURES:")
    for f in combined["failed_details"]:
        log(f"    - {f['label']}: status={f['status']} elapsed={f['elapsed_s']}s error={f['error']}")

if combined["status_502"] > 0 or combined["status_0"] > 0:
    log(f"\n  ❌ PRODUCTION ISSUE DETECTED: {combined['failed']} failures")
else:
    log(f"\n  ✅ ALL PRODUCTION TESTS PASSED")

# Save results
out_path = os.path.join(os.path.dirname(__file__), "..", "production_test_results.json")
with open(out_path, "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
log(f"\n  Results saved to: {out_path}")
log("Done.")
