import argparse
import json
import os
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory


def build_model(num_classes: int, img_size: int = 224):
    base = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    # Stronger head to help separation across classes
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(data_dir: str, out_dir: str = "models", epochs: int = 5, batch_size: int = 32, img_size: int = 224):
    data_dir = os.path.abspath(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.1,
        subset="training",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.1,
        subset="validation",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )

    # Get class names BEFORE any dataset mapping/augmentation
    class_names = train_ds.class_names
    num_classes = len(class_names)
    with open(out_path / "labels.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    # Basic augmentation pipeline (mild but helpful)
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.1),
    ])
    AUTOTUNE = None
    try:
        AUTOTUNE = keras.utils.AUTOTUNE
    except Exception:
        pass
    def apply_aug(x, y):
        return aug(x, training=True), y
    train_ds = train_ds.map(apply_aug)

    # Performance: cache and prefetch
    if AUTOTUNE is not None:
        train_ds = train_ds.prefetch(AUTOTUNE)
        val_ds = val_ds.prefetch(AUTOTUNE)

    # Compute class weights to mitigate imbalance
    # Count samples per class directly from directory structure
    counts = []
    for cname in class_names:
        class_dir = Path(data_dir) / cname
        # Count files recursively under each class folder
        n = sum(1 for p in class_dir.rglob("*") if p.is_file())
        counts.append(max(n, 1))
    total = sum(counts)
    class_weight = {i: float(total) / (len(class_names) * counts[i]) for i in range(len(class_names))}

    model = build_model(num_classes, img_size)
    callbacks = [
        # Keep an .h5 checkpoint for broad compatibility
        keras.callbacks.ModelCheckpoint(str(out_path / "plant_disease_model_tf.h5"), monitor="val_accuracy", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, class_weight=class_weight)

    # Optional fine-tuning: unfreeze top of the base model and train a bit longer
    try:
        base = None
        for layer in model.layers:
            if isinstance(layer, keras.Model) and layer.name.startswith('mobilenetv2'):
                base = layer
                break
        if base is not None:
            base.trainable = True
            for l in base.layers[:-30]:
                l.trainable = False
            model.compile(optimizer=keras.optimizers.Adam(1e-4),
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"]) 
            model.fit(train_ds, validation_data=val_ds, epochs=max(1, epochs//4), callbacks=callbacks, class_weight=class_weight)
    except Exception:
        pass
    # Save using the recommended Keras format
    model.save(out_path / "plant_disease_model_tf.keras")
    # Optionally export a SavedModel for serving/TFLite usage
    try:
        model.export(out_path / "plant_disease_model_tf_export")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="PlantVillage")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    train(args.data_dir, args.out_dir, args.epochs, args.batch_size, args.img_size)
