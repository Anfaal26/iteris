"""
BRISC tumor-type classifier — glioma / meningioma / pituitary / non_tumor.

The segmentation models (AttentionResUNet / LiteUNet, see models.py) only
predict tumor-vs-background; they have no concept of tumor type. BRISC's
filename convention already encodes the type per image (see
ingestion.build_brisc_dicts), so this trains a small from-scratch CNN
classifier directly on that label — no extra annotation needed.

Deliberately NOT using an ImageNet-pretrained backbone: those expect 3-channel
RGB with ImageNet normalisation, which would fight the project's existing
1-channel z-score preprocessing (transforms.py / configs/brisc.yaml) and break
train/serve consistency with the FastAPI backend (server/app/inference.py).
A compact from-scratch CNN keeps the same preprocessing everywhere.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .ingestion import build_brisc_dicts
from .splits import patient_level_split
from .utils import count_parameters, get_device, seed_all

CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'non_tumor']
IMAGE_SIZE = 256


class TumorClassifier(nn.Module):
    """Small from-scratch CNN — 5 conv blocks + global average pool + FC.

    ~1.2M params, fast to train on a single Kaggle T4, no pretrained-weight
    normalisation mismatch with the rest of the pipeline.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = len(CLASS_NAMES)):
        super().__init__()

        def block(ic, oc, stride=2):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(in_channels, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
            block(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class BriscClassificationDataset(Dataset):
    """Loads (image, tumor_type) pairs with the same preprocessing as
    inference.py's segmentation path — greyscale, resize, z-score over
    nonzero pixels — so a single normalisation convention covers both models.
    """

    def __init__(self, records: list[dict], image_size: int = IMAGE_SIZE):
        self.records = [r for r in records if r.get('tumor_type', 'unknown') != 'unknown']
        self.image_size = image_size
        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(rec['image']).convert('L').resize(
            (self.image_size, self.image_size), Image.BILINEAR,
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        nz = arr[arr != 0]
        if nz.size > 0:
            mean, std = float(nz.mean()), float(nz.std())
        else:
            mean, std = float(arr.mean()), float(arr.std())
        arr = (arr - mean) / (std + 1e-6)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        label = self.class_to_idx[rec['tumor_type']]
        return tensor, label


def run_classifier_training(cfg: dict) -> dict:
    """Trains the tumor-type classifier and returns history/checkpoint paths,
    mirroring the shape of iteris.training.run_training's return dict.
    """
    seed_all(cfg.get('seed', 42))
    device = get_device()

    records = build_brisc_dicts(cfg['data_root'])
    train_recs, val_recs, test_recs = patient_level_split(
        records,
        val_split=cfg.get('val_split', 0.15),
        test_split=cfg.get('test_split', 0.10),
        label_frac=cfg.get('label_frac', 1.0),
        seed=cfg.get('seed', 42),
    )

    train_ds = BriscClassificationDataset(train_recs)
    val_ds = BriscClassificationDataset(val_recs)
    test_ds = BriscClassificationDataset(test_recs)
    print(f'[classifier] usable (labelled) samples — train: {len(train_ds)}  '
          f'val: {len(val_ds)}  test: {len(test_ds)}')

    batch_size = cfg.get('batch_size', 32)
    workers = cfg.get('dataloader_workers', 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    model = TumorClassifier().to(device)
    print(f'[classifier] TumorClassifier params: {count_parameters(model):,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 1e-3),
                                  weight_decay=cfg.get('weight_decay', 1e-5))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    checkpoint_dir = Path(cfg['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'brisc_tumor_classifier_best.pt'

    epochs = cfg.get('epochs', 30)
    patience = cfg.get('patience', 8)
    best_val_acc = 0.0
    epochs_without_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= max(len(train_ds), 1)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                val_loss += loss.item() * images.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
        val_loss /= max(len(val_ds), 1)
        val_acc = correct / max(len(val_ds), 1)
        scheduler.step(val_acc)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - t0
        print(f'[epoch {epoch:02d}/{epochs}] train_loss={train_loss:.4f} '
              f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({elapsed:.1f}s)')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improve = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'class_names': CLASS_NAMES,
                        'val_acc': val_acc,
                        'epoch': epoch}, checkpoint_path)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f'[classifier] early stop at epoch {epoch} (no improvement for {patience} epochs)')
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])

    return {
        'model': model,
        'history': history,
        'best_val_acc': best_val_acc,
        'checkpoint': str(checkpoint_path),
        'test_loader': test_loader,
        'class_names': CLASS_NAMES,
    }


def evaluate_classifier_test_set(model: nn.Module, test_loader: DataLoader, cfg: dict) -> dict:
    """Per-class precision/recall/F1 + overall accuracy on the held-out test set."""
    device = get_device()
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    n_classes = len(CLASS_NAMES)
    per_class = {}
    for c in range(n_classes):
        tp = int(((all_preds == c) & (all_labels == c)).sum())
        fp = int(((all_preds == c) & (all_labels != c)).sum())
        fn = int(((all_preds != c) & (all_labels == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[CLASS_NAMES[c]] = {'precision': round(precision, 4),
                                      'recall': round(recall, 4), 'f1': round(f1, 4)}

    accuracy = float((all_preds == all_labels).mean()) if len(all_labels) > 0 else 0.0
    return {'accuracy': round(accuracy, 4), 'per_class': per_class, 'n_test': len(all_labels)}


def save_classifier_summary_json(history: dict, test_metrics: dict, cfg: dict, best_val_acc: float) -> Path:
    checkpoint_dir = Path(cfg['checkpoint_dir'])
    summary = {
        'best_val_acc': round(best_val_acc, 4),
        'test_metrics': test_metrics,
        'epochs_trained': len(history['train_loss']),
        'class_names': CLASS_NAMES,
        'config': {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool, list))},
    }
    out_path = checkpoint_dir / 'brisc_tumor_classifier_summary.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print(f'[classifier] summary written to {out_path}')
    return out_path
