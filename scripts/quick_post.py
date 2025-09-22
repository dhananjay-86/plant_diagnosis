import os, json, requests, glob
PORT = int(os.environ.get("PORT", "5008"))
BASE = f"http://127.0.0.1:{PORT}"
print("Health:", requests.get(BASE+"/health").status_code)
print("Model info:", requests.get(BASE+"/model_info").json())

# find a few images
roots = [
  r"PlantVillage/Pepper__bell___Bacterial_spot",
  r"PlantVillage/Potato___healthy",
  r"PlantVillage/Tomato__Tomato_mosaic_virus"
]
imgs = []
for r in roots:
  imgs += glob.glob(r + "/**/*.JPG", recursive=True)[:1]

for p in imgs:
  with open(p, 'rb') as f:
    files = {"image": (os.path.basename(p), f, "image/jpeg")}
    data = {"location": json.dumps({"lat":12.3, "lon":45.6})}
    res = requests.post(BASE+"/analyze", files=files, data=data)
    print("\n=>", p)
    print(res.status_code)
    try:
      print(json.dumps(res.json(), indent=2)[:600])
    except Exception:
      print(res.text[:600])
