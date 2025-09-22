# AI Plant Diagnostics (Flask + TensorFlow)

A web app that detects plant diseases from leaf images. Built with Flask (backend), TensorFlow/Keras (MobileNetV2 transfer learning), and a simple HTML/CSS/JS frontend. Trained on the PlantVillage dataset.

## Features
- Upload image or capture from camera, then Run AI Analysis
- Auto-captures geolocation (lat/lon) on image select/capture
- TensorFlow model (.keras preferred) loaded lazily on first request
- Diagnostics endpoints: `/health`, `/labels`, `/model_info`, `/geocode`, `/reverse_geocode`

## Quickstart
Requirements: Python 3.11, PowerShell (Windows). A virtual environment `.venv-tf` is recommended.

```powershell
# Optional: create/activate venv in PowerShell
# python -m venv .venv-tf
# & ".\.venv-tf\Scripts\Activate.ps1"

# Install deps
pip install -r requirements.txt

# Run server
$env:PORT="5031"; $env:DEBUG="0"; python app.py
# open http://127.0.0.1:5031/
```

## Training
Place the PlantVillage dataset in `PlantVillage/` with class subfolders (as in the standard dataset). Then:

```powershell
python training/train_tf.py --data_dir "PlantVillage" --out_dir "models" --epochs 8 --batch_size 32 --img_size 224
```

Artifacts:
- `models/labels.json`
- `models/plant_disease_model_tf.keras` (preferred)
- `models/plant_disease_model_tf.h5` (best checkpoint)

## Evaluation
```powershell
python scripts/eval_tf.py
```

## Deployment (local production)
A production runner using `waitress` is provided:

```powershell
pip install waitress
$env:PORT="8080"; python deploy.py
```

## Notes
- The dataset `PlantVillage/` is ignored by git via `.gitignore`.
- Set `OPENWEATHER_API_KEY` to enable OpenWeatherMap geocoding.