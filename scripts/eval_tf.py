import os
import json
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory


def main(root_dir: str = ".", data_subdir: str = "PlantVillage", models_subdir: str = "models", img_size: int = 224, batch_size: int = 32):
    root = Path(root_dir)
    data_dir = root / data_subdir
    models_dir = root / models_subdir
    labels_path = models_dir / "labels.json"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json not found at {labels_path}")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    num_classes = len(labels)

    model_path = models_dir / "plant_disease_model_tf.keras"
    if not model_path.exists():
        model_path = models_dir / "plant_disease_model_tf.h5"
    if not model_path.exists():
        raise FileNotFoundError("No TF model found (.keras or .h5)")

    print("Loading model from", model_path)
    model = keras.models.load_model(model_path, compile=False)

    val_ds = image_dataset_from_directory(
        str(data_dir),
        validation_split=0.1,
        subset="validation",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )

    # Evaluate accuracy and per-class stats
    correct = 0
    count = 0
    per_class_total = [0] * num_classes
    per_class_correct = [0] * num_classes

    for images, targets in val_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        t = targets.numpy()
        correct += int((preds == t).sum())
        count += len(t)
        for i in range(len(t)):
            idx = int(t[i])
            per_class_total[idx] += 1
            if preds[i] == t[i]:
                per_class_correct[idx] += 1

    acc = correct / count if count else 0.0
    print(f"Validation accuracy: {acc:.4f} ({correct}/{count})")

    per_class_acc = [
        (labels[i], (per_class_correct[i] / per_class_total[i] if per_class_total[i] else 0.0), per_class_total[i])
        for i in range(num_classes)
    ]
    per_class_acc.sort(key=lambda x: x[1])
    print("Per-class accuracy (worst 5):")
    for name, a, n in per_class_acc[:5]:
        print(f"  {name:45s}  acc={a:.3f}  n={n}")
    print("Per-class accuracy (best 5):")
    for name, a, n in per_class_acc[-5:]:
        print(f"  {name:45s}  acc={a:.3f}  n={n}")


if __name__ == "__main__":
    main(root_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
