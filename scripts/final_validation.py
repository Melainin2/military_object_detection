"""Quick validation: verify config + soutenance edge cases via API."""
import requests, os

BASE = "http://localhost:8000"

# Test known military image to verify config
with open("test_images_military_vs_civilian/mil_f35_jet.jpg", "rb") as f:
    r = requests.post(f"{BASE}/predict", files={"file": ("test.jpg", f.read(), "image/jpeg")}, timeout=30)
    data = r.json()
    print(f"F-35 jet: {data['message'][:60]} | det={len(data['detections'])}")
    for d in data["detections"]:
        print(f"  {d['class_name']}: {d['confidence']:.4f}")

# Test a hard FN that should now pass (helicopter)
with open("test_images_military_vs_civilian/mil_helicopter_02.jpg", "rb") as f:
    r = requests.post(f"{BASE}/predict", files={"file": ("test.jpg", f.read(), "image/jpeg")}, timeout=30)
    data = r.json()
    print(f"Helicopter: {data['message'][:60]} | det={len(data['detections'])}")
    for d in data["detections"]:
        print(f"  {d['class_name']}: {d['confidence']:.4f}")

# Test a known FP (should now be rejected with correct threshold)
with open("test_images_military_vs_civilian/civ_abstract_01.jpg", "rb") as f:
    r = requests.post(f"{BASE}/predict", files={"file": ("test.jpg", f.read(), "image/jpeg")}, timeout=30)
    data = r.json()
    print(f"Abstract: {data['message'][:60]} | det={len(data['detections'])}")

# Soutenance edge cases
edge_cases = [
    ("sout_blurred_soldier", "military"),
    ("sout_civilian_airliner", "civilian"),
    ("sout_civilian_ship", "civilian"),
    ("sout_toy_tank", "civilian"),
    ("sout_videogame", "civilian"),
    ("sout_military_parade", "military"),
    ("sout_low_light", "military"),
    ("sout_military_poster", "military"),
    ("sout_hunter_rifle", "civilian"),
    ("sout_police_officer", "civilian"),
    ("sout_woman_traditional", "civilian"),
]

print("\nSoutenance edge cases:")
correct = 0
for label, expected in edge_cases:
    path = f"test_images_military_vs_civilian/{label}.jpg"
    if not os.path.exists(path):
        continue
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/predict", files={"file": ("test.jpg", f.read(), "image/jpeg")}, timeout=30)
        data = r.json()
    vit_passed = data.get("message", "") != "No military object detected"
    is_correct = (expected == "civilian" and not vit_passed) or (expected == "military" and vit_passed)
    if is_correct:
        correct += 1
    status = "PASS" if is_correct else "FAIL"
    print(f"  [{status}] {label:30s} expected={expected:10s} actual={'MIL' if vit_passed else 'CIV'}")

print(f"\nSoutenance: {correct}/{len(edge_cases)} correct ({(correct/len(edge_cases)*100):.0f}%)")
