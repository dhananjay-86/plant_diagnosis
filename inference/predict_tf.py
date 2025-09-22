import json
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers


def _preprocess(image: Image.Image, img_size: int = 224):
    img = image.convert("RGB").resize((img_size, img_size))
    # IMPORTANT: Do NOT apply preprocess_input here because the model
    # already includes a keras.applications.mobilenet_v2.preprocess_input layer.
    # Just pass raw float32 pixels; the model graph will handle preprocessing.
    x = np.asarray(img).astype("float32")
    x = np.expand_dims(x, 0)
    return x


def _build_model(num_classes: int, img_size: int = 224):
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    # Compilation is optional for inference, but set for completeness
    model.compile(optimizer=keras.optimizers.Adam(3e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def load_model_tf(model_dir_or_h5: str, num_classes: int | None = None):
    """
    Try loading a TF model. Supports:
    - Keras .keras SavedModel
    - SavedModel directory
    - H5 full model
    - H5 weights-only (if num_classes is provided, rebuilds architecture and loads weights)
    """
    # First, try standard loading
    try:
        return keras.models.load_model(model_dir_or_h5)
    except Exception:
        # Retry with compile=False for broader compatibility
        try:
            return keras.models.load_model(model_dir_or_h5, compile=False)
        except Exception:
            pass

    # If still failing and it's an .h5 file, attempt weights-only path
    if model_dir_or_h5.lower().endswith('.h5') and num_classes is not None:
        model = _build_model(num_classes)
        try:
            model.load_weights(model_dir_or_h5)
            return model
        except Exception as e:
            raise e

    # Give up with a clear error
    raise RuntimeError(f"Failed to load TensorFlow model from {model_dir_or_h5}")


def predict_image_tf(model, image: Image.Image, labels: List[str], topk: int = 3, temperature: float = 1.0) -> Tuple[int, str, float, List[Dict]]:
    x = _preprocess(image)
    probs = model.predict(x, verbose=0)[0]
    # Safety: if model outputs logits (shouldn't, but just in case), apply softmax
    if probs.ndim == 1 and (probs < 0).any() and (probs > 1).any():
        e = np.exp(probs - np.max(probs))
        probs = e / e.sum()
    # Optional temperature scaling to reduce peaky distributions
    if temperature and temperature != 1.0:
        logits = np.log(np.clip(probs, 1e-8, 1.0))
        logits = logits / temperature
        e = np.exp(logits - np.max(logits))
        probs = e / e.sum()
    idx = np.argsort(-probs)[: min(topk, len(labels))]
    conf = probs[idx]
    top_list = [{"label": labels[i], "confidence": float(conf[j])} for j, i in enumerate(idx)]
    best = int(idx[0])
    return best, labels[best], float(probs[best]), top_list
