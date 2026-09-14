"""Evaluation pipeline for SignalScope model on standard and unseen-generator test sets."""

import os
import json
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
)

from model.config import (
    DEVICE,
    BATCH_SIZE,
    OPERATING_THRESHOLD,
    MODEL_WEIGHTS_PATH,
    MODEL_METADATA_PATH,
    EVALUATION_RESULTS_PATH,
    RESULTS_DIR,
    RANDOM_SEED,
    NUM_WORKERS,
)
from model.dataset import get_dataset_splits
from model.model import build_model


def evaluate_loader(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    threshold: float = OPERATING_THRESHOLD,
) -> Dict[str, Any]:
    """Runs evaluation over a PyTorch DataLoader and computes all required SIH metrics.

    Metrics computed strictly without fabrication:
    - ROC-AUC
    - Macro-F1
    - Confusion Matrix (TN, FP, FN, TP)
    - Accuracy
    - Precision
    - Recall
    - False Positive Rate (FPR) at operating threshold

    Returns:
        dict: Complete dictionary of evaluated metrics and curve points.
    """
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            all_targets.extend(labels.numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)

    # Core required metrics
    auc = float(roc_auc_score(y_true, y_prob))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = map(int, cm.ravel())

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "roc_auc": round(auc, 4),
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "operating_threshold": threshold,
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "confusion_matrix": {
            "true_negatives_real": tn,
            "false_positives_ai": fp,
            "false_negatives_real": fn,
            "true_positives_ai": tp,
            "raw_matrix": [[tn, fp], [fn, tp]],
        },
        "num_samples": len(y_true),
        "class_distribution": {
            "real": int(np.sum(y_true == 0)),
            "ai_generated": int(np.sum(y_true == 1)),
        },
    }


def run_full_evaluation(
    max_samples_per_class: Optional[int] = None,
    weights_path: str = str(MODEL_WEIGHTS_PATH),
) -> Dict[str, Any]:
    """Evaluates trained checkpoint on both standard and unseen-generator test splits.

    Saves results cleanly to model/results/evaluation_metrics.json.
    """
    print(f"Loading trained weights from: {weights_path}", flush=True)
    model = build_model(pretrained=False)
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.", flush=True)
    else:
        print(" Warning: Model weights file not found. Running baseline un-finetuned evaluation.", flush=True)

    model = model.to(DEVICE)

    # 1. Get test datasets
    splits = get_dataset_splits(seed=RANDOM_SEED, max_samples_per_class=max_samples_per_class)
    test_loader = DataLoader(splits["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    unseen_loader = DataLoader(splits["unseen_test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print("Evaluating Standard Test Set...", flush=True)
    standard_metrics = evaluate_loader(model, test_loader, DEVICE)

    print("Evaluating Unseen-Generator Held-Out Test Set...", flush=True)
    unseen_metrics = evaluate_loader(model, unseen_loader, DEVICE)


    combined_results = {
        "evaluation_status": "success",
        "model_architecture": "ResNet-18",
        "standard_test_metrics": standard_metrics,
        "unseen_generator_test_metrics": unseen_metrics,
        "summary": {
            "standard_roc_auc": standard_metrics["roc_auc"],
            "unseen_generator_roc_auc": unseen_metrics["roc_auc"],
            "standard_macro_f1": standard_metrics["macro_f1"],
            "unseen_generator_macro_f1": unseen_metrics["macro_f1"],
            "standard_accuracy": standard_metrics["accuracy"],
            "unseen_generator_accuracy": unseen_metrics["accuracy"],
        }
    }

    # Save to file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(EVALUATION_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)

    print(f" Full evaluation results saved to: {EVALUATION_RESULTS_PATH}")
    print("\n" + "=" * 60)
    print("FINAL EVALUATION METRICS SUMMARY:")
    print(f" Standard Test Set ROC-AUC:           {standard_metrics['roc_auc']:.4f}")
    print(f" Standard Test Set Macro-F1:          {standard_metrics['macro_f1']:.4f}")
    print(f" Standard Test Set Accuracy:          {standard_metrics['accuracy']:.4f}")
    print(f" Unseen-Generator Test Set ROC-AUC:  {unseen_metrics['roc_auc']:.4f}")
    print(f" Unseen-Generator Test Set Macro-F1: {unseen_metrics['macro_f1']:.4f}")
    print(f" Unseen-Generator Test Set Accuracy: {unseen_metrics['accuracy']:.4f}")
    print("=" * 60 + "\n")

    return combined_results


if __name__ == "__main__":
    run_full_evaluation()
