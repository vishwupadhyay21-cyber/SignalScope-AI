"""Data preprocessing and augmentation pipelines for SignalScope ML."""

from torchvision import transforms
from model.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


def get_train_transforms() -> transforms.Compose:
    """Returns PyTorch torchvision transformations for model training with data augmentations."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Returns PyTorch torchvision transformations for validation, testing, and evaluation."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def get_inference_transforms() -> transforms.Compose:
    """Returns PyTorch torchvision transformations for backend inference, strictly matching preprocessor.py."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
