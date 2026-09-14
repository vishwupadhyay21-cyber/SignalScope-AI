"""CIFAR-10 PyTorch Dataset Loader for SignalScope Training."""

import os
import pickle
import numpy as np
from typing import List, Tuple

CIFAR10_DIR = r"D:\cifar-10-python\cifar-10-batches-py"


def load_cifar10_real_images(
    cifar_dir: str = CIFAR10_DIR, max_samples: int = 50000
) -> List[Tuple[np.ndarray, int]]:
    """Loads raw 32x32 RGB numpy array images from CIFAR-10 batches as Real (label=0).

    Includes all 5 training batches plus test_batch for up to 60k real images.
    """
    samples = []
    if not os.path.exists(cifar_dir):
        return samples

    batch_files = [
        "data_batch_1", "data_batch_2", "data_batch_3",
        "data_batch_4", "data_batch_5", "test_batch",
    ]
    for b_name in batch_files:
        b_path = os.path.join(cifar_dir, b_name)
        if os.path.exists(b_path):
            with open(b_path, "rb") as f:
                entry = pickle.load(f, encoding="bytes")
                raw_data = entry[b"data"]
                # Reshape N x 3072 -> N x 3 x 32 x 32 -> N x 32 x 32 x 3 (HWC RGB)
                reshaped = raw_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                for img_array in reshaped:
                    samples.append((img_array, 0))  # Label 0 = Real
                    if len(samples) >= max_samples:
                        return samples
    return samples
