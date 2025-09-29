import io
import json
import os
from datetime import datetime
from typing import Tuple

from flask import Flask, jsonify, render_template, request
from PIL import Image


def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")

    # Lazy-load model at first request to speed up initial import
    app.model = None
    app.labels = None
    app.model_kind = None
    app.last_model_load_error = None

    def ensure_model_loaded():
        if app.model is None:
            labels_path = os.path.join("models", "labels.json")
            tf_dir = os.path.join("models", "plant_disease_model_tf_export")
            tf_h5 = os.path.join("models", "plant_disease_model_tf.h5")
            tf_keras = os.path.join("models", "plant_disease_model_tf.keras")
            torch_pth = os.path.join("models", "plant_disease_model.pth")

            if os.path.exists(labels_path):
                with open(labels_path, "r", encoding="utf-8") as f:
                    app.labels = json.load(f)
            else:
                app.labels = None

            # Load TensorFlow model if available
            if app.labels is not None and (os.path.exists(tf_dir) or os.path.exists(tf_h5) or os.path.exists(tf_keras)):
                try:
                    from inference.predict_tf import load_model_tf  # type: ignore
                    # Prefer .keras -> .h5 -> exported dir
                    model_path = tf_keras if os.path.exists(tf_keras) else (tf_h5 if os.path.exists(tf_h5) else tf_dir)
                    app.model = load_model_tf(model_path, num_classes=len(app.labels))
                    app.model_kind = "tf"
                    app.last_model_load_error = None
                    return
                except Exception as e:
                    app.model = None
                    app.model_kind = None
                    # Record the error for diagnostics
                    try:
                        app.last_model_load_error = str(e)
                    except Exception:
                        app.last_model_load_error = "Unknown TF load error"

            # Fallback: Load PyTorch model if available
            if app.labels is not None and os.path.exists(torch_pth):
                try:
                    from inference.predict import load_model as load_model_torch  # type: ignore
                    app.model = load_model_torch(torch_pth, num_classes=len(app.labels))
                    app.model_kind = "torch"
                    app.last_model_load_error = None
                    return
                except Exception as e:
                    app.model = None
                    app.model_kind = None
                    try:
                        app.last_model_load_error = str(e)
                    except Exception:
                        app.last_model_load_error = "Unknown Torch load error"

            # No model available
            app.model = None
            app.model_kind = None
            app.last_model_load_error = None

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "time": datetime.utcnow().isoformat() + "Z"})

    @app.get("/labels")
    def get_labels():
        ensure_model_loaded()
        if app.labels is None:
            return jsonify({"labels": [], "message": "Model not trained yet."})
        return jsonify({"labels": app.labels})

    @app.get("/model_info")
    def model_info():
        ensure_model_loaded()
        model_kind = getattr(app, "model_kind", None)
        available = {
            "tf_h5": os.path.exists(os.path.join("models", "plant_disease_model_tf.h5")),
            "tf_keras": os.path.exists(os.path.join("models", "plant_disease_model_tf.keras")),
            "tf_export": os.path.exists(os.path.join("models", "plant_disease_model_tf_export")),
            "torch_pth": os.path.exists(os.path.join("models", "plant_disease_model.pth")),
        }
        return jsonify({
            "loaded": app.model is not None,
            "model_kind": model_kind,
            "labels_loaded": app.labels is not None,
            "available": available,
            "last_model_load_error": getattr(app, "last_model_load_error", None),
        })

    @app.post("/analyze")
    def analyze():
        ensure_model_loaded()

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded."}), 400
        f = request.files["image"]
        if f.filename == "":
            return jsonify({"error": "Empty filename."}), 400

        location = request.form.get("location")  # JSON string with lat/lon
        address = request.form.get("address")

        img_bytes = f.read()
        try:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            return jsonify({"error": "Invalid image."}), 400

        if app.model is None or app.labels is None:
            # Graceful response when model isn't trained yet
            return jsonify({
                "message": "Model not available yet. Train the model first.",
                "receivedLocation": json.loads(location) if location else None,
                "address": address,
            }), 503

        # Dispatch to the correct predictor based on the loaded model kind
        if app.model_kind == "tf":
            from inference.predict_tf import predict_image_tf  # type: ignore
            pred_idx, pred_label, confidence, topk = predict_image_tf(app.model, image, app.labels)
        elif app.model_kind == "torch":
            from inference.predict import predict_image  # type: ignore
            pred_idx, pred_label, confidence, topk = predict_image(app.model, image, app.labels)
        else:
            return jsonify({
                "message": "Model not available yet. Train the model first.",
                "receivedLocation": json.loads(location) if location else None,
                "address": address,
            }), 503
        diagnosis = diagnosis_from_label(pred_label)

        return jsonify({
            "prediction": pred_label,
            "confidence": round(float(confidence), 4),
            "topk": topk,
            "diagnosis": diagnosis,
            "receivedLocation": json.loads(location) if location else None,
            "address": address,
        })

    @app.get("/reverse_geocode")
    def reverse_geocode():
        """Reverse geocode coordinates into a human-readable place.

        If OPENWEATHER_API_KEY is set, use OpenWeatherMap Reverse Geocoding.
        Otherwise, fall back to OpenStreetMap Nominatim.
        """
        import requests
        lat = request.args.get("lat")
        lon = request.args.get("lon")
        if not lat or not lon:
            return jsonify({"error": "lat and lon required"}), 400

        api_key = os.environ.get("OPENWEATHER_API_KEY")
        try:
            if api_key:
                # OpenWeatherMap Reverse Geocoding
                url = (
                    "https://api.openweathermap.org/geo/1.0/reverse?lat="
                    + str(lat)
                    + "&lon="
                    + str(lon)
                    + "&limit=1&appid="
                    + api_key
                )
                r = requests.get(url, timeout=10)
                arr = r.json() if r.ok else []
                if isinstance(arr, list) and arr:
                    item = arr[0]
                    name = item.get("name")
                    state = item.get("state")
                    country = item.get("country")
                    display = ", ".join([x for x in [name, state, country] if x])
                    return jsonify({"address": display, "raw": item})
                # If empty, continue to OSM fallback below

            # Fallback: OpenStreetMap Nominatim
            osm_url = (
                "https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat="
                + str(lat)
                + "&lon="
                + str(lon)
            )
            r = requests.get(osm_url, headers={"User-Agent": "plant-diagnostics-app/1.0"}, timeout=10)
            data = r.json()
            return jsonify({
                "address": data.get("display_name"),
                "raw": data,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/geocode")
    def geocode():
        """Direct geocoding: city/state/country name -> coordinates using OpenWeatherMap.

        Query params:
        - q: free text (e.g., "London, UK" or "City, State, Country")
        - limit: optional (default 1)
        Requires OPENWEATHER_API_KEY in environment.
        """
        import requests
        q = request.args.get("q")
        limit = int(request.args.get("limit", "1"))
        if not q:
            return jsonify({"error": "query 'q' is required"}), 400
        api_key = os.environ.get("OPENWEATHER_API_KEY")
        if not api_key:
            return jsonify({"error": "OPENWEATHER_API_KEY not set on server"}), 500
        try:
            url = (
                "https://api.openweathermap.org/geo/1.0/direct?q="
                + requests.utils.quote(q)
                + f"&limit={limit}&appid="
                + api_key
            )
            r = requests.get(url, timeout=10)
            if not r.ok:
                return jsonify({"error": f"OpenWeather API error {r.status_code}"}), r.status_code
            arr = r.json()
            # Normalize to minimal shape
            results = [
                {
                    "name": it.get("name"),
                    "lat": it.get("lat"),
                    "lon": it.get("lon"),
                    "state": it.get("state"),
                    "country": it.get("country"),
                }
                for it in (arr if isinstance(arr, list) else [])
            ]
            return jsonify({"results": results})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def diagnosis_from_label(label: str) -> str:
    # Minimal built-in guidance. Can be extended.
    advice = {
        "Tomato___Late_blight": "Likely Phytophthora infestans. Remove infected leaves, apply copper-based fungicide, avoid overhead watering.",
        "Tomato___Early_blight": "Alternaria solani suspected. Prune lower leaves, rotate crops, use chlorothalonil or copper.",
        "Tomato___healthy": "Leaf appears healthy. Maintain good watering and nutrition.",
        "Pepper__bell___Bacterial_spot": "Bacterial spot: remove infected tissue, sanitize tools, consider copper sprays.",
        "Potato___Early_blight": "Early blight: remove debris, rotate crops, apply appropriate fungicide.",
        "Potato___Late_blight": "Late blight: urgent removal and disposal of infected plants; protect with fungicides.",
    }
    # Default advice
    return advice.get(label, "Maintain proper sanitation, avoid leaf wetness, and consider consulting local extension services.")


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("DEBUG", "1") == "1"
    app.run(host="127.0.0.1", port=port, debug=debug)
