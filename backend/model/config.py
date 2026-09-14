"""Configuration file for SignalScope ML Model Training, Evaluation, and Inference."""

import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Dataset configuration
DATASET_DIR = Path(os.environ.get("SIGNALSCOPE_DATASET_DIR", r"D:\SIH_gradcam\train"))

REAL_DIR = DATASET_DIR / "REAL"

AI_DIR = DATASET_DIR / "FAKE"

# Output directories
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_WEIGHTS_PATH = CHECKPOINT_DIR / "best_model.pth"
MODEL_METADATA_PATH = CHECKPOINT_DIR / "model_metadata.json"
EVALUATION_RESULTS_PATH = RESULTS_DIR / "evaluation_metrics.json"
ROBUSTNESS_RESULTS_PATH = RESULTS_DIR / "robustness_analysis.json"

# Image processing configuration (strictly compatible with app.ml.preprocessor)
IMAGE_SIZE = (224, 224)
INPUT_CHANNELS = 3

# Normalization parameters (ImageNet standard)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Class mappings
CLASS_MAPPING = {
    0: "Real",
    1: "AI-generated"
}
REVERSE_CLASS_MAPPING = {
    "Real": 0,
    "AI-generated": 1
}

# Training Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
CLASSIFIER_DROPOUT = 0.3

# Fast training: limit samples per class to speed up CPU training
# Set to None to use full dataset (slow on CPU)
MAX_SAMPLES_PER_CLASS = 5000

# Calibration & Threshold Configuration
OPERATING_THRESHOLD = 0.35  # Calibrated decision threshold for generalisation across unseen generators (SDXL/Diffusion)
UNCERTAINTY_MARGIN = 0.15   # Confidence margin around threshold for "Likely AI" verdict hedging




# Hardware configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2 if os.name != 'nt' else 0
