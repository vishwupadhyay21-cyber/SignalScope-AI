"""Backup of model/config.py"""
import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATASET_DIR = Path(os.environ.get("SIGNALSCOPE_DATASET_DIR", r"D:\my_dataset"))
REAL_DIR = DATASET_DIR / "real"
AI_DIR = DATASET_DIR / "AI-generated"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_WEIGHTS_PATH = CHECKPOINT_DIR / "best_model.pth"
MODEL_METADATA_PATH = CHECKPOINT_DIR / "model_metadata.json"
EVALUATION_RESULTS_PATH = RESULTS_DIR / "evaluation_metrics.json"
ROBUSTNESS_RESULTS_PATH = RESULTS_DIR / "robustness_analysis.json"

IMAGE_SIZE = (224, 224)
INPUT_CHANNELS = 3

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

CLASS_MAPPING = {
    0: "Real",
    1: "AI-generated"
}
REVERSE_CLASS_MAPPING = {
    "Real": 0,
    "AI-generated": 1
}

RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
CLASSIFIER_DROPOUT = 0.3

MAX_SAMPLES_PER_CLASS = 5000

OPERATING_THRESHOLD = 0.50
UNCERTAINTY_MARGIN = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2 if os.name != 'nt' else 0
