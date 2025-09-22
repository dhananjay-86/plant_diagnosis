import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_transforms(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_model(model_path: str, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def predict_image(model, image: Image.Image, labels: List[str], topk: int = 3) -> Tuple[int, str, float, List[Dict]]:
    device = next(model.parameters()).device
    tfm = _build_transforms()
    x = tfm(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        conf, idx = probs.topk(min(topk, len(labels)), dim=1)
    conf = conf.squeeze(0).tolist()
    idx = idx.squeeze(0).tolist()
    top_list = [
        {"label": labels[i], "confidence": float(c)} for i, c in zip(idx, conf)
    ]
    best_idx = idx[0]
    return best_idx, labels[best_idx], float(conf[0]), top_list
