import os
import sys
import json
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("PORT", "5006"))
BASE = f"http://127.0.0.1:{PORT}"

samples = [
    ROOT / "PlantVillage" / "Pepper__bell___Bacterial_spot",
    ROOT / "PlantVillage" / "Potato___healthy",
    ROOT / "PlantVillage" / "Tomato__Tomato_mosaic_virus",
]

# Pick one image from each class
images = []
for d in samples:
    if d.exists() and d.is_dir():
        # find first jpg
        for p in d.glob("**/*.JPG"):
            images.append(p)
            break

if not images:
    print("No sample images found to test.")
    sys.exit(1)

# Check health
try:
    h = requests.get(f"{BASE}/health", timeout=5)
    print("Health:", h.status_code, h.text[:120])
except Exception as e:
    print("Server not reachable:", e)
    sys.exit(1)

for img in images:
    with open(img, "rb") as f:
        files = {"image": (img.name, f, "image/jpeg")}
        data = {"location": json.dumps({"lat": 12.34, "lon": 56.78})}
        r = requests.post(f"{BASE}/analyze", files=files, data=data, timeout=20)
        print("\n==>", img)
        print(r.status_code)
        try:
            print(json.dumps(r.json(), indent=2)[:800])
        except Exception:
            print(r.text[:800])
        time.sleep(0.5)
