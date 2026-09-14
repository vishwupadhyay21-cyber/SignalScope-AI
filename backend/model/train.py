"""Training pipeline for SignalScope Real vs AI-Generated image classifier."""

import os
import json
import time
import random
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from model.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    RANDOM_SEED,
    MODEL_WEIGHTS_PATH,
    MODEL_METADATA_PATH,
    CLASS_MAPPING,
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_WORKERS,
)
from model.dataset import get_dataset_splits
from model.model import build_model


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Sets random seeds for reproducibility across random, numpy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    """Runs a single training epoch and returns (mean_loss, epoch_accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels, _) in enumerate(dataloader, 1):
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.50).long()
        correct += (preds == labels.long()).sum().item()
        total += inputs.size(0)

        if batch_idx % 15 == 0 or batch_idx == len(dataloader):
            print(f"  [Batch {batch_idx:03d}/{len(dataloader):03d}] Loss: {loss.item():.4f} | Acc: {correct/total:.4f}", flush=True)


    mean_loss = total_loss / total
    accuracy = correct / total
    return mean_loss, accuracy


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float, float]:
    """Evaluates the model on validation data and returns (mean_loss, accuracy, roc_auc)."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(logits)

            all_targets.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    total = len(all_targets)
    mean_loss = total_loss / total
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    preds = (all_probs >= 0.50).astype(int)
    accuracy = float(np.mean(preds == all_targets))

    # ROC-AUC requires at least 2 distinct classes (0 and 1) in target array
    if len(np.unique(all_targets)) >= 2:
        try:
            auc = float(roc_auc_score(all_targets, all_probs))
            if np.isnan(auc):
                auc = accuracy
        except Exception:
            auc = accuracy
    else:
        # Single-class dataset fallback (e.g. only Real images uploaded)
        auc = accuracy

    return mean_loss, accuracy, auc


def train_model(
    max_samples_per_class: Optional[int] = None,
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
) -> Dict[str, Any]:
    """Executes complete training and validation routine, saving best checkpoint.

    Args:
        max_samples_per_class: Optional limit on samples per class (useful for rapid compute budgets).
        epochs: Total number of training epochs.
        batch_size: DataLoader batch size.
        lr: Learning rate.

    Returns:
        dict: Summary of best achieved metrics and training stats.
    """
    set_seed(RANDOM_SEED)
    print(f"Starting SignalScope Training on Device: {DEVICE}", flush=True)

    # 1. Prepare Datasets & Loaders
    splits = get_dataset_splits(seed=RANDOM_SEED, max_samples_per_class=max_samples_per_class)
    train_dataset = splits["train"]
    val_dataset = splits["val"]

    print(f"Train Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    # 2. Build Model, Criterion, Optimizer, Scheduler
    model = build_model(pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = 0.0
    best_epoch = 0
    history = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_auc = validate_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        ep_duration = time.time() - ep_start
        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{ep_duration:.1f}s] | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, ROC-AUC: {val_auc:.4f}",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
        })

        # Save checkpoint if best ROC-AUC
        if val_auc > best_val_auc or epoch == 1:
            best_val_auc = val_auc
            best_epoch = epoch

            # Save state dict
            os.makedirs(os.path.dirname(MODEL_WEIGHTS_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)

            # Save metadata
            metadata = {
                "architecture": "ResNet-18",
                "version": "1.0.0",
                "class_mapping": CLASS_MAPPING,
                "input_size": IMAGE_SIZE,
                "normalization": {
                    "mean": NORMALIZE_MEAN,
                    "std": NORMALIZE_STD,
                },
                "best_epoch": best_epoch,
                "best_val_auc": best_val_auc,
                "training_device": DEVICE,
                "random_seed": RANDOM_SEED,
            }
            with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            print(f" Saved new best checkpoint to {MODEL_WEIGHTS_PATH} (Val AUC: {val_auc:.4f})", flush=True)

    total_time = time.time() - start_time
    print(f"Training Complete in {total_time/60:.2f} mins. Best Epoch: {best_epoch} with Val AUC: {best_val_auc:.4f}", flush=True)


    return {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "total_time_seconds": total_time,
        "history": history,
    }


if __name__ == "__main__":
    train_model()
