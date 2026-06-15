"""
Compare API vs direct classifier results for key test images.
"""
import subprocess, sys, os, time, requests, cv2, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.classifier_service import ViTClassifier

BASE = "http://localhost:8000"
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Start server
print("Starting server...")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    env={**os.environ, "PYTHONPATH": os.getcwd()}
)

for _ in range(60):
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        if r.json().get("vit_loaded"):
            break
    except:
        pass
    time.sleep(2)
print("Server ready")

# Test images covering all categories
test_images = [
    "mil_f35_jet.jpg",
    "mil_abrams_tank.jpg",
    "mil_tank_desert.jpg",
    "mil_apache_01.jpg",
    "mil_helicopter_02.jpg",
    "mil_f18_01.jpg",
    "mil_warship_01.jpg",
    "mil_artillery_01.jpg",
    "civ_hunter_004.jpg",
    "civ_city_001.jpg",
    "civ_moto_01.jpg",
    "sout_civilian_ship.jpg",
    "sout_blurred_soldier.jpg",
    "sout_military_poster.jpg",
]

# Direct classifier
c = ViTClassifier(threshold=0.05, top_k=20)
c.load()
print(f"Direct: threshold={c.threshold}, top_k={c.top_k}, mil_classes={len(c.military_class_ids)}")
print()

header = f"{'Image':30s} {'API':6s} {'Direct':6s} {'API score':10s} {'Dir score':10s} Top-1"
print(header)
print("-" * 90)

differences = []

for name in test_images:
    path = os.path.join("test_images_military_vs_civilian", name)
    if not os.path.exists(path):
        print(f"{name:30s} FILE NOT FOUND")
        continue
    
    img = cv2.imread(path)
    if img is None:
        continue
    
    # Direct classifier
    direct_result = c.predict(img)
    dir_mil = direct_result["is_military"]
    dir_score = direct_result["military_score"]
    dir_top1 = direct_result["top_predictions"][0]["class"][:35] if direct_result["top_predictions"] else "none"
    
    # API
    with open(path, "rb") as f:
        img_bytes = f.read()
    r = requests.post(f"{BASE}/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")}, timeout=30)
    api_data = r.json()
    api_mil = api_data.get("message", "") != "No military object detected"
    api_score = api_data.get("vit_military_score", 0)
    
    match = "SAME" if api_mil == dir_mil else "DIFF"
    if match == "DIFF":
        differences.append((name, api_mil, dir_mil, api_score, dir_score))
    
    print(f"{name:30s} {'MIL' if api_mil else 'CIV':6s} {'MIL' if dir_mil else 'CIV':6s} {api_score:.4f}    {dir_score:.4f}    {dir_top1} [{match}]")

print()
if differences:
    print(f"DIFFERENCES FOUND: {len(differences)}")
    for name, api_mil, dir_mil, api_score, dir_score in differences:
        print(f"  {name}: API={'MIL' if api_mil else 'CIV'} Direct={'MIL' if dir_mil else 'CIV'} (scores: API={api_score:.4f} Direct={dir_score:.4f})")
else:
    print("ALL RESULTS MATCH between API and Direct classifier")

print(f"\nChecking ViTClassifier config from server...")
r = requests.get(f"{BASE}/health", timeout=5)
print(f"  Health: {r.json()}")

proc.terminate()
try:
    proc.wait(timeout=10)
except:
    proc.kill()
