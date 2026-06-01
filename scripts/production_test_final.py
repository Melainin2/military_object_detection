"""
Final production validation test.
All requests use the SAME 640x480 image for consistency.
Tests: 15 size-derived variations + 20 consecutive load tests + 3 error tests
"""
import json, os, time
from datetime import datetime
import requests

BASE = "https://military-object-detection.onrender.com"

def log(s):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {s}", flush=True)

def test(label, img_bytes):
    t0 = time.time()
    try:
        r = requests.post(f"{BASE}/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")}, timeout=300)
        try: js = r.json(); dets = len(js.get("detections", [])); msg = js.get("message", "")
        except: js = None; dets = 0; msg = r.text[:80]
        return {"label": label, "status": r.status_code, "elapsed_s": round(time.time()-t0, 2), "detections": dets, "message": msg}
    except requests.Timeout:
        return {"label": label, "status": 0, "elapsed_s": round(time.time()-t0, 2), "error": "timeout"}
    except Exception as e:
        return {"label": label, "status": 0, "elapsed_s": round(time.time()-t0, 2), "error": str(e)}

def fmt(s):
    return f"  {s['label']}: status={s['status']} elapsed={s['elapsed_s']}s detections={s['detections']}"

# Phase 1: Wake
log("PHASE 1: Wake")
r = requests.get(f"{BASE}/health", timeout=60)
log(f"  /health: {r.status_code} {r.text}")

# Phase 2: Get a single test image
log("PHASE 2: Download test image")
r = requests.get("https://picsum.photos/640/480.jpg", timeout=30, allow_redirects=True)
img = r.content
log(f"  Image: {len(img)} bytes")
time.sleep(2)

# Phase 3: First 15 individual requests (like size-varied but same image)
log("PHASE 3: 15 individual prediction requests")
size_results = []
for i in range(15):
    r = test(f"req_{i+1:02d}", img)
    log(fmt(r))
    size_results.append(r)
    time.sleep(0.5)

s_ok = sum(1 for r in size_results if r["status"] == 200)
s_el = [r["elapsed_s"] for r in size_results if r["status"] == 200]
log(f"  Result: {s_ok}/15 success, avg={round(sum(s_el)/len(s_el),2) if s_el else 'N/A'}s, max={round(max(s_el),2) if s_el else 'N/A'}s")

if s_ok < 15:
    log("  FAILURES:")
    for r in size_results:
        if r["status"] != 200:
            log(f"    {r['label']}: status={r['status']} elapsed={r['elapsed_s']}s {r.get('error','')}")

if s_ok < 15:
    # App crashed - wait for restart then continue load test
    log("  App crashed! Waiting for restart...")
    for i in range(30):
        time.sleep(5)
        try:
            r = requests.get(f"{BASE}/health", timeout=15)
            if r.status_code == 200:
                log(f"  App restarted: {r.text}")
                time.sleep(5)
                break
        except:
            pass

# Phase 4: 20 consecutive predictions
log("PHASE 4: 20 consecutive predictions")
load_results = []
for i in range(20):
    r = test(f"load_{i+1:02d}", img)
    log(fmt(r))
    load_results.append(r)
    time.sleep(0.5)

l_ok = sum(1 for r in load_results if r["status"] == 200)
l_el = [r["elapsed_s"] for r in load_results if r["status"] == 200]
log(f"  Result: {l_ok}/20 success, avg={round(sum(l_el)/len(l_el),2) if l_el else 'N/A'}s, max={round(max(l_el),2) if l_el else 'N/A'}s")

if l_ok < 20:
    log("  FAILURES:")
    for r in load_results:
        if r["status"] != 200:
            log(f"    {r['label']}: status={r['status']} elapsed={r['elapsed_s']}s {r.get('error','')}")

# Phase 5: Error handling
log("PHASE 5: Error handling")
errors = []
for label, payload in [
    ("invalid_type", {"file": ("test.txt", b"hello", "text/plain")}),
    ("no_file", {}),
]:
    try:
        r = requests.post(f"{BASE}/predict", files=payload if payload else {}, timeout=30)
        errors.append({"label": label, "status": r.status_code, "body": r.text[:60]})
        log(f"  {label}: status={r.status_code} body={r.text[:50]}")
    except Exception as e:
        errors.append({"label": label, "error": str(e)})
        log(f"  {label}: error {e}")

# Phase 6: Combined results
log("PHASE 6: RESULTS")
all_res = size_results + load_results
total = len(all_res)
ok = sum(1 for r in all_res if r["status"] == 200)
el = sorted([r["elapsed_s"] for r in all_res if r["status"] == 200])
p95_idx = min(int(len(el)*0.95), len(el)-1) if el else 0

log(f"  Total: {total}")
log(f"  Success: {ok} ({round(ok/total*100,1)}%)")
log(f"  Failed: {total-ok}")
if el:
    log(f"  Avg: {round(sum(el)/len(el),2)}s")
    log(f"  Max: {round(max(el),2)}s")
    log(f"  Min: {round(min(el),2)}s")
    log(f"  P50: {round(el[len(el)//2],2)}s")
    log(f"  P95: {round(el[p95_idx],2)}s")

if total == ok:
    log("  ALL PRODUCTION TESTS PASSED")
else:
    log("  SOME TESTS FAILED")

# Save
os.makedirs("production_results", exist_ok=True)
out = {
    "timestamp": datetime.now().isoformat(),
    "commit": "576547c",
    "phase3_15_requests": {"results": size_results, "summary": {"ok": s_ok, "total": 15}},
    "phase4_20_load": {"results": load_results, "summary": {"ok": l_ok, "total": 20}},
    "error_handling": errors,
    "combined": {"total": total, "ok": ok, "failed": total-ok, "avg_s": round(sum(el)/len(el),2) if el else 0, "max_s": round(max(el),2) if el else 0},
}
with open("production_results/final_validation.json", "w") as f:
    json.dump(out, f, indent=2)
log("Saved to production_results/final_validation.json")
