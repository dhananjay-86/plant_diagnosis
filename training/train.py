import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm


def get_dataloaders(data_dir: str, img_size: int = 224, batch_size: int = 32, val_split: float = 0.1):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=tfm)
    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return dataset, train_loader, val_loader


def train(data_dir: str, out_dir: str = "models", epochs: int = 5, lr: float = 3e-4, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)
    num_classes = len(dataset.classes)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        running_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
        train_loss = running_loss / total
        train_acc = correct / total if total else 0

        # Validation
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += x.size(0)
        val_acc = correct / total if total else 0

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_path / "plant_disease_model.pth")

    # Save labels
    with open(out_path / "labels.json", "w", encoding="utf-8") as f:
        json.dump(dataset.classes, f, indent=2)

    print(f"Training complete. Best val acc: {best_acc:.4f}. Model and labels saved to {out_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="PlantVillage", help="Path to PlantVillage dataset root")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    train(args.data_dir, args.out_dir, args.epochs, args.lr, args.batch_size)
