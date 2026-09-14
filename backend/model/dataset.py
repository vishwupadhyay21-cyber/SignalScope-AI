"""Dataset loader and split manager for SignalScope Real vs AI-Generated images."""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

from model.config import REAL_DIR, AI_DIR, RANDOM_SEED
from model.transforms import get_train_transforms, get_eval_transforms


class SignalScopeDataset(Dataset):
    """PyTorch Dataset for Real vs AI-Generated Image Forensics (file-path based)."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    tensor = self.transform(img)
                else:
                    from torchvision import transforms as T
                    tensor = T.ToTensor()(img)
                return tensor, label, path
        except Exception:
            return torch.zeros((3, 224, 224), dtype=torch.float32), label, path


class CifarDataset(Dataset):
    """PyTorch Dataset for CIFAR-10 numpy images (used as Real class samples)."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_array, label = self.samples[idx]
        try:
            img = Image.fromarray(img_array.astype(np.uint8), mode="RGB")
            if self.transform:
                tensor = self.transform(img)
            else:
                from torchvision import transforms as T
                tensor = T.ToTensor()(img)
            return tensor, label, f"cifar10_idx_{idx}"
        except Exception:
            return torch.zeros((3, 224, 224), dtype=torch.float32), label, f"cifar10_idx_{idx}"


def collect_dataset_samples(real_dir=REAL_DIR, ai_dir=AI_DIR):
    """Scans dataset directories (recursively) and returns sample lists for Real and AI-generated classes."""
    real_samples = []
    ai_samples = []

    # Determine potential base directories to search
    search_dirs = []
    if os.path.exists(real_dir):
        search_dirs.append(real_dir)
    if os.path.exists(ai_dir):
        search_dirs.append(ai_dir)

    base_dir = Path(real_dir).parent
    if os.path.exists(base_dir):
        search_dirs.append(base_dir)
    ai_keywords = ["fake", "synth", "dalle", "midjourney", "flux"]
    real_keywords = ["real", "auth", "genuin", "orig"]

    seen_files = set()
    seen_files = set()

    for s_dir in search_dirs:
        for root, _, files in os.walk(s_dir):
            folder_lower = root.lower().replace("\\", "/")
            # Determine class based on path keywords
            is_ai   = any(k in folder_lower for k in ai_keywords)
            is_real = any(k in folder_lower for k in real_keywords) and not is_ai

            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    full_path = os.path.join(root, fname)
                    if full_path in seen_files:
                        continue
                    seen_files.add(full_path)

                    if is_ai:
                        ai_samples.append((full_path, 1))
                    elif is_real:
                        real_samples.append((full_path, 0))

    return real_samples, ai_samples


def collect_cifar10_samples(max_samples=50000):
    """Loads CIFAR-10 images as numpy arrays to use as Real class samples."""
    from model.load_cifar_dataset import load_cifar10_real_images, CIFAR10_DIR
    if not os.path.exists(CIFAR10_DIR):
        print(f"  [CIFAR-10] Not found at {CIFAR10_DIR} — skipping.")
        return []
    samples = load_cifar10_real_images(cifar_dir=CIFAR10_DIR, max_samples=max_samples)
    print(f"  [CIFAR-10] Loaded {len(samples)} real images from {CIFAR10_DIR}")
    return samples


def get_dataset_splits(seed=RANDOM_SEED, max_samples_per_class=None):
    """Generates leak-free Train / Val / Test / Unseen-Generator datasets.

    CIFAR-10 images (all labeled Real=0) are loaded from
    D:\\cifar-10-python\\cifar-10-batches-py and merged with any file-based
    real images via ConcatDataset.

    Returns:
        dict with keys: train, val, test, unseen_test
    """
    from torch.utils.data import ConcatDataset

    real_samples, ai_samples = collect_dataset_samples()

    cifar_max = max_samples_per_class if max_samples_per_class else 50000
    cifar_samples = collect_cifar10_samples(max_samples=cifar_max)

    if max_samples_per_class:
        real_samples = real_samples[:max_samples_per_class]
        ai_samples   = ai_samples[:max_samples_per_class]

    rng = np.random.RandomState(seed)
    rng.shuffle(real_samples)
    rng.shuffle(ai_samples)
    if cifar_samples:
        rng.shuffle(cifar_samples)

    print(
        f"  [Dataset] File-based real: {len(real_samples)} | "
        f"AI: {len(ai_samples)} | CIFAR-10 real: {len(cifar_samples)}"
    )

    # Hold out 10% of AI as unseen generator split
    n_ai = len(ai_samples)
    unseen_ai   = ai_samples[:int(0.10 * n_ai)]
    seen_ai     = ai_samples[int(0.10 * n_ai):]

    # Hold out 5% of file-based real as unseen split
    n_real = len(real_samples)
    unseen_real = real_samples[:int(0.05 * n_real)]
    seen_real   = real_samples[int(0.05 * n_real):]

    unseen_test = unseen_real + unseen_ai
    rng.shuffle(unseen_test)

    # 80 / 10 / 10 split for file-based
    r_train = int(0.80 * len(seen_real))
    r_val   = int(0.90 * len(seen_real))
    a_train = int(0.80 * len(seen_ai))
    a_val   = int(0.90 * len(seen_ai))

    # 80 / 10 / 10 split for CIFAR-10
    c_train = int(0.80 * len(cifar_samples))
    c_val   = int(0.90 * len(cifar_samples))

    file_train = seen_real[:r_train] + seen_ai[:a_train]
    file_val   = seen_real[r_train:r_val] + seen_ai[a_train:a_val]
    file_test  = seen_real[r_val:] + seen_ai[a_val:]
    rng.shuffle(file_train); rng.shuffle(file_val); rng.shuffle(file_test)

    cifar_train = cifar_samples[:c_train]
    cifar_val   = cifar_samples[c_train:c_val]
    cifar_test  = cifar_samples[c_val:]

    train_tf = get_train_transforms()
    eval_tf  = get_eval_transforms()

    def _make(file_samps, cifar_samps, tf):
        parts = []
        if file_samps:
            parts.append(SignalScopeDataset(file_samps, transform=tf))
        if cifar_samps:
            parts.append(CifarDataset(cifar_samps, transform=tf))
        if len(parts) == 2:
            return ConcatDataset(parts)
        if len(parts) == 1:
            return parts[0]
        return SignalScopeDataset([], transform=tf)

    return {
        "train":       _make(file_train, cifar_train, train_tf),
        "val":         _make(file_val,   cifar_val,   eval_tf),
        "test":        _make(file_test,  cifar_test,  eval_tf),
        "unseen_test": SignalScopeDataset(unseen_test, transform=eval_tf),
    }
