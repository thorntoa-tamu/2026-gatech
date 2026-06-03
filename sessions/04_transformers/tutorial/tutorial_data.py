"""Shared dataset helpers for the ML4FP tutorial notebooks."""

from __future__ import annotations

import os

import numpy as np
import requests
import torch


JETCLASS_URL = "https://hqu.web.cern.ch/datasets/JetClass/example/JetClass_example_100k.root"
JETCLASS_FILENAME = "JetClass_example_100k.root"

BINARY_CLASSES = (0, 8)
BINARY_NAMES = ("QCD", "top")


def ensure_jetclass_example(data_dir: str = "../data") -> str:
    """Download the JetClass example file if it is not already present."""
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, JETCLASS_FILENAME)

    if os.path.exists(data_path):
        print(f"Dataset already exists at {data_path}")
        return data_path

    print(f"Downloading JetClass example dataset from {JETCLASS_URL} ...")
    response = requests.get(JETCLASS_URL, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(data_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB", end="")

    if total > 0:
        print()
    print("Done!")
    return data_path


def prepare_small_binary_task(
    x_particles: np.ndarray,
    y: np.ndarray,
    *,
    n_use: int = 6000,
    seed: int = 42,
) -> dict[str, object]:
    """Prepare the shared QCD-vs-top task used in Notebooks 1 and 2."""
    x_all = x_particles.transpose(0, 2, 1).astype("float32")
    mask_all = np.any(x_all != 0, axis=2)
    labels10 = y.argmax(axis=1)

    sel = np.isin(labels10, BINARY_CLASSES)
    x_bin = x_all[sel].copy()
    mask_bin = mask_all[sel]
    y_bin = (labels10[sel] == BINARY_CLASSES[1]).astype(np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_bin))
    x_bin = x_bin[perm]
    mask_bin = mask_bin[perm]
    y_bin = y_bin[perm]

    n_use = min(n_use, len(x_bin))
    n_train = int(0.8 * n_use)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_use)

    # Match the kinematic preprocessing used in the first notebook.
    eps = 1e-6
    x_bin[..., 0] = np.log(np.clip(x_bin[..., 0], eps, None))
    x_bin[..., 3] = np.log(np.clip(x_bin[..., 3], eps, None))

    real = mask_bin[train_idx]
    feats = x_bin[train_idx][real]
    mu = feats.mean(0)
    sd = feats.std(0) + 1e-6

    x_bin = (x_bin - mu) / sd
    x_bin = x_bin * mask_bin[..., None]

    return {
        "binary_classes": list(BINARY_CLASSES),
        "binary_names": list(BINARY_NAMES),
        "x_train": torch.tensor(x_bin[train_idx]),
        "mask_train": torch.tensor(mask_bin[train_idx], dtype=torch.bool),
        "y_train": torch.tensor(y_bin[train_idx], dtype=torch.long),
        "x_test": torch.tensor(x_bin[test_idx]),
        "mask_test": torch.tensor(mask_bin[test_idx], dtype=torch.bool),
        "y_test": torch.tensor(y_bin[test_idx], dtype=torch.long),
        "mean": mu,
        "std": sd,
    }
