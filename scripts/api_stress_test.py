"""
API stress test: 100+ mixed images through the full ViT+YOLO pipeline.
Starts server, runs tests, validates metrics, generates report.
"""
import subprocess, sys, os, time, json, requests, signal, threading, cv2, glob
from collections import Counter

BASE = "http://localhost:8000"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ground truth
mislabeled = {"mil_air_02.jpg", "mil_aircraft_01.jpg", "mil_tank_05.jpg"}
soutenance_gt = {
    "sout_woman_traditional": "civilian", "sout_civilian_airliner": "civilian",
    "sout_civilian_ship": "civilian", "sout_toy_tank": "civilian",
    "sout_videogame": "civilian", "sout_military_parade": "military",
    "sout_blurred_soldier": "military", "sout_low_light": "military",
    "sout_military_poster": "military", "sout_hunter_rifle": "civilian",
    "sout_police_officer": "civilian",
}

def get_gt(name):
    if name.startswith("mil_"):
        return "civilian" if name in mislabeled else "military"
    if name.startswith("civ_"): return "civilian"
    if name.startswith("sout_"):
        return soutenance_gt.get(name.replace(".jpg",""), "unknown")
    return "unknown"

def wait_for_server(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{BASE}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get("model_loaded") and data.get("vit_loaded"):
                    return True
                time.sleep(3)
        except:
            time.sleep(2)
    return False

# =============================================================
print("=" * 70)
print("API STRESS TEST")
print("=" * 70)

# Start server
print("\nStarting server...")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    env={**os.environ, "PYTHONPATH": os.getcwd()}
)
print(f"Server PID: {proc.pid}")

if not wait_for_server():
    print("FAILED to start server!")
    proc.kill()
    sys.exit(1)
print("Server ready.\n")

# =============================================================
print("Running API tests on 100+ images...")
# =============================================================
# Select: all military images (57) + sample of civilian images (~60) = ~117 total
all_files = sorted(glob.glob(os.path.join("test_images_military_vs_civilian", "*.jpg")))
military_files = [f for f in all_files if get_gt(os.path.basename(f)) == "military"]
civilian_files = [f for f in all_files if get_gt(os.path.basename(f)) == "civilian"]
sout_files = [f for f in all_files if get_gt(os.path.basename(f)) == "unknown"]

# Test ALL military + ALL soutenance + sample of civilian
test_files = military_files + sout_files
# Add 60 civilian images
sample_size = min(60, len(civilian_files))
test_files += civilian_files[:sample_size]

print(f"  Military: {len(military_files)}, Civilian: {sample_size}, Soutenance: {len(sout_files)}")
print(f"  Total: {len(test_files)}")

results = []
failures = []
start_ts = time.time()

for i, filepath in enumerate(test_files):
    name = os.path.basename(filepath)
    gt = get_gt(name)
    
    with open(filepath, "rb") as f:
        img_bytes = f.read()
    
    t0 = time.time()
    try:
        r = requests.post(f"{BASE}/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")}, timeout=120)
        elapsed = time.time() - t0
    except Exception as e:
        results.append({"file": name, "ground_truth": gt, "error": str(e)[:100], "api_time_s": round(time.time()-t0, 2)})
        continue
    
    if r.status_code != 200:
        results.append({"file": name, "ground_truth": gt, "error": f"HTTP {r.status_code}", "api_time_s": round(elapsed, 2)})
        continue
    
    data = r.json()
    vit_passed = data.get("message", "") != "No military object detected"
    detections = data.get("detections", [])
    
    record = {
        "file": name, "ground_truth": gt,
        "vit_passed": vit_passed, "detections": len(detections),
        "message": data.get("message", ""),
        "vit_confidence": data.get("vit_confidence", 0),
        "vit_military_score": data.get("vit_military_score", 0),
        "api_time_s": round(elapsed, 2),
    }
    if detections:
        record["detection_classes"] = [d["class_name"] for d in detections]
        record["detection_confs"] = [d["confidence"] for d in detections]
        # Verify boxes are in original image space
        img = cv2.imread(filepath)
        if img is not None:
            h, w = img.shape[:2]
            for det in detections:
                box = det["box"]
                if not (0 <= box[0] <= w and 0 <= box[1] <= h and 0 <= box[2] <= w and 0 <= box[3] <= h):
                    record["box_error"] = f"box={box} outside {w}x{h}"
    
    results.append(record)
    
    is_fp = (gt == "civilian" and vit_passed)
    is_fn = (gt == "military" and not vit_passed)
    if is_fp or is_fn:
        failures.append({
            "file": name, "ground_truth": gt,
            "type": "FP" if is_fp else "FN",
            "vit_passed": vit_passed, "detections": len(detections),
            "message": data.get("message", ""), "time_s": round(elapsed, 2),
        })
    
    # Progress
    if (i+1) % 20 == 0 or (i+1) == len(test_files):
        fp_now = sum(1 for r in results if r.get("ground_truth") == "civilian" and r.get("vit_passed"))
        fn_now = sum(1 for r in results if r.get("ground_truth") == "military" and not r.get("vit_passed"))
        print(f"  [{i+1}/{len(test_files)}] FPs={fp_now} FNs={fn_now} ({time.time()-start_ts:.0f}s)")

total_time = time.time() - start_ts
print(f"\nCompleted {len(results)} tests in {total_time:.0f}s")

# =============================================================
print("\n" + "=" * 70)
print("METRICS")
print("=" * 70)

known = [r for r in results if r.get("ground_truth") in ("civilian", "military")]
tp = sum(1 for r in known if r["ground_truth"] == "military" and r.get("vit_passed", False))
tn = sum(1 for r in known if r["ground_truth"] == "civilian" and not r.get("vit_passed", False))
fp = sum(1 for r in known if r["ground_truth"] == "civilian" and r.get("vit_passed", False))
fn = sum(1 for r in known if r["ground_truth"] == "military" and not r.get("vit_passed", False))

print(f"\n  Confusion Matrix ({len(known)} known):")
print(f"  {'':30s} Pred CIV    Pred MIL")
print(f"  {'Actual CIV':30s} {tn:10d} {fp:10d}")
print(f"  {'Actual MIL':30s} {fn:10d} {tp:10d}")

accuracy = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else 0
precision = tp/(tp+fp) if (tp+fp)>0 else 0
recall = tp/(tp+fn) if (tp+fn)>0 else 0
f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
specificity = tn/(tn+fp) if (tn+fp)>0 else 0

print(f"\n  Metrics:")
print(f"  Accuracy:    {accuracy:.4f}  ({tp+tn}/{tp+tn+fp+fn})")
print(f"  Precision:   {precision:.4f}  ({tp}/{tp+fp})")
print(f"  Recall:      {recall:.4f}  ({tp}/{tp+fn})")
print(f"  Specificity: {specificity:.4f}  ({tn}/{tn+fp})")
print(f"  F1 Score:    {f1:.4f}")
print(f"  Avg time:    {sum(r.get('api_time_s',0) for r in known)/len(known):.2f}s")

# YOLO detection rate
mil_passed = [r for r in known if r["ground_truth"] == "military" and r.get("vit_passed", False)]
yolo_detected = sum(1 for r in mil_passed if r.get("detections", 0) > 0)
print(f"\n  YOLO on passed military: {yolo_detected}/{len(mil_passed)} ({yolo_detected/max(len(mil_passed),1)*100:.1f}%)")

# Check for box errors
box_errors = [r for r in results if r.get("box_error")]
if box_errors:
    print(f"\n  BOX ERRORS: {len(box_errors)}")
    for be in box_errors:
        print(f"    {be['file']}: {be.get('box_error','')}")
else:
    print(f"\n  Box coordinate validation: ALL PASSED (no out-of-bounds boxes)")

# =============================================================
print(f"\n{'='*70}")
print(f"STRESS TEST COMPLETE: {len(results)} images, {(time.time()-start_ts)/60:.1f} minutes")
print(f"{'='*70}")

# Save results
os.makedirs("scripts", exist_ok=True)
metrics = {
    "total_api_tested": len(results),
    "known_ground_truth": len(known),
    "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    "vit_gatekeeper": {
        "accuracy": round(accuracy,4), "precision": round(precision,4),
        "recall": round(recall,4), "specificity": round(specificity,4), "f1_score": round(f1,4),
    },
    "yolo_detection": {"passed_vit": len(mil_passed), "detected": yolo_detected},
    "total_time_s": round(total_time, 1),
    "avg_time_s": round(total_time/max(len(results),1), 2),
    "box_errors": len(box_errors),
}
with open("scripts/api_stress_results.json", "w") as f:
    json.dump({"metrics": metrics, "results": results, "failures": failures}, f, indent=2)
print(f"Results saved to scripts/api_stress_results.json")

# Cleanup
proc.terminate()
try:
    proc.wait(timeout=10)
except:
    proc.kill()
print("Server stopped.")
