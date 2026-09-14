# SignalScope ML Model Module

This directory contains the machine learning classification engine, dataset loaders, training pipeline, evaluation protocols, visual explainability (Grad-CAM), robustness benchmarking, and inference adapter for **SignalScope (SIH 2026)**.

---

## Directory Structure

```
model/
├── config.py             # Global hyperparameters, data paths, class mappings, device configuration
├── transforms.py         # Torchvision preprocessing (224x224 RGB) and training augmentations
├── dataset.py            # PyTorch Dataset loader & leak-free generator-aware train/val/test splits
├── model.py              # ResNet-18 Transfer Learning classifier with custom classification head
├── train.py              # Training loop with BCE loss, AdamW optimizer, and best-AUC checkpointing
├── evaluate.py           # Evaluation pipeline computing ROC-AUC, Macro-F1, Confusion Matrix, FPR
├── explain.py            # Grad-CAM visual explainability and activation map generator (Module A)
├── robustness.py         # Robustness benchmark across JPEG compression, scale, and noise (Module C & G)
├── provenance.py         # C2PA / Content Credentials & EXIF forensic metadata parser (Module D)
├── multimodal.py         # Image + text caption consistency verification engine (Module E)
├── inference.py          # PyTorch SignalScopeDetector implementing FastAPI BaseDetector contract
├── checkpoints/          # Saved model weights (best_model.pth) and metadata JSON
└── results/              # Evaluated metrics JSON and visual analysis plots
```

---

## Quickstart Instructions

### 1. Training
To train the model on the `D:\my_dataset` dataset:
```bash
python model/train.py
```
This trains the model, validates on each epoch, and saves the best checkpoint to `model/checkpoints/best_model.pth`.

### 2. Evaluation
To run full metric evaluation on standard and unseen-generator held-out test sets:
```bash
python model/evaluate.py
```
Results will be saved to `model/results/evaluation_metrics.json`.

### 3. Robustness Benchmarking
To benchmark performance against JPEG compression, spatial resizing, and Gaussian noise:
```bash
python model/robustness.py
```
Results will be saved to `model/results/robustness_analysis.json`.

### 4. Integration Verification
Run existing backend unit tests to verify backend integration:
```bash
python -m unittest discover -s tests
```
