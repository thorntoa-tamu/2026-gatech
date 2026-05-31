import torch
import h5py
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
import requests
import re
import os
from urllib.parse import urljoin
import numpy as np
from pathlib import Path


def collate_point_cloud(batch, max_part=150):
    """
    Collate function for point clouds and labels with truncation performed per batch.

    Args:
        batch (list of dicts): Each element is a dictionary with keys:
            - "X" (Tensor): Point cloud of shape (N, F)
            - "y" (Tensor): Label tensor
            - "cond" (optional, Tensor): Conditional info
            - "pid" (optional, Tensor): Particle IDs
            - "add_info" (optional, Tensor): Extra features

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing collated tensors:
            - "X": (B, M, F) Truncated point clouds
            - "y": (B, num_classes)
            - "cond", "pid", "add_info" (optional, shape (B, M, ...))
    """
    batch_X = [item["X"] for item in batch]
    batch_y = [item["y"] for item in batch]

    # Stack once to avoid repeated slicing
    point_clouds = torch.stack(batch_X)  # (B, N, F)
    labels = torch.stack(batch_y)  # (B, num_classes)

    # Use validity mask based on feature index 2
    valid_mask = point_clouds[:, :, 2] != 0
    max_particles = min(valid_mask.sum(dim=1).max().item(), max_part)

    # Truncate point clouds
    truncated_X = point_clouds[:, :max_particles, :]  # (B, M, F)
    result = {"X": truncated_X, "y": labels}
    return result

class HEPDataset(Dataset):
    def __init__(
        self,
        file_path,            
        num_evt = -1,
        num_part = 30,
    ):
        """
        Args:
            file_paths (list): List of file paths.
        """
        self.file_path = file_path
        self.num_evt = num_evt
        self.num_part = num_part
        self._file_cache = self._get_file(self.file_path)

    def __len__(self):
        if self.num_evt > 0:
            return self.num_evt
        else:
            return len(self._file_cache['data'])

    def _get_file(self, file_path):
        self._file_cache = h5py.File(file_path, "r")
        return self._file_cache

    def __getitem__(self, idx):
        sample = {}

        sample["X"] = torch.tensor(self._file_cache["data"][idx][:self.num_part], dtype=torch.float32)
        sample["y"] = torch.tensor(self._file_cache["pid"][idx], dtype=torch.int64)
        return sample

    def __del__(self):
        # Clean up: close all cached file handles.
        for f in self._file_cache.values():
            try:
                f.close()
            except Exception as e:
                print(f"Error closing file: {e}")


def load_data(
    dataset_name,
    path,
    batch=100,
    dataset_type="train",
    num_evt = -1,
    num_workers=16,
    rank=0,
    size=1,
):
    names = [dataset_name]
    types = [dataset_type]

    dataset_path = os.path.join(path, dataset_name, dataset_type)
    h5_file = list(Path(dataset_path).glob("*.h5"))[0]

    data = HEPDataset(h5_file,num_evt)

    loader = DataLoader(
        data,
        batch_size=batch,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
        sampler=None,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_point_cloud,
    )
    return loader


# Pixel cluster data (BIB tutorial)

CLUSTER_FEATURE_KEYS = [
    'cluster_energy', 'cluster_time', 'cluster_r', 'incident_angle',
    'cluster_size_tot', 'cluster_size_x', 'cluster_size_y',
    'cluster_rms_x', 'cluster_rms_y',
    'cluster_skew_x', 'cluster_skew_y',
    'cluster_aspect', 'cluster_ecc',
]

class PixelClusterDataset(Dataset):
    def __init__(
        self,
        file_path,
        num_evt=-1,
        max_hits=9,
        label_override=None,
        raw_features=("energy", "time", "x", "y"),
    ):
        self.file_path = file_path
        self.max_hits = max_hits
        self.label_override = label_override
        self.raw_features = tuple(raw_features)

        self._file = h5py.File(file_path, "r")
        self._n = len(self._file["raw_hits/energy"])
        if num_evt > 0:
            self._n = min(self._n, num_evt)

        print(f"  Pre-loading {self._n} samples from {os.path.basename(file_path)}...")
        print(f"  Raw features: {self.raw_features}")

        rh = self._file["raw_hits"]

        arrays = {
            name: rh[name][:self._n]
            for name in self.raw_features
        }

        self._X = torch.zeros(
            self._n,
            max_hits,
            len(self.raw_features),
            dtype=torch.float32,
        )

        for i in range(self._n):
            e = arrays["energy"][i]
            nhits = min(len(e), max_hits)

            if nhits > 0:
                order = np.argsort(-e)[:nhits]

                for j, name in enumerate(self.raw_features):
                    vals = arrays[name][i][order].astype(np.float32)
                    self._X[i, :nhits, j] = torch.from_numpy(vals)

        del arrays

        if label_override is not None:
            self._labels = torch.full((self._n,), label_override, dtype=torch.int64)
        else:
            self._labels = torch.from_numpy(
                np.array(self._file["truth_label"][:self._n]).astype(np.int64)
            )

        self._file.close()
        print(f"  Done. Tensor shape: {self._X.shape}")

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return {"X": self._X[idx], "y": self._labels[idx]}

    def cluster_features(self, keys=None):
        with h5py.File(self.file_path, "r") as f:
            if keys is None:
                keys = list(f["clusters"].keys())
            return {k: np.array(f["clusters"][k][:self._n]) for k in keys}

    def labels(self):
        if self.label_override is not None:
            return np.full(self._n, self.label_override, dtype=np.int8)
        return self._labels.numpy().astype(np.int8)

    def raw_hit_counts(self):
        with h5py.File(self.file_path, "r") as f:
            if "clusters" in f and "cluster_size_tot" in f["clusters"]:
                return np.array(f["clusters/cluster_size_tot"][:self._n])
        return (self._X[:, :, 0] != 0).sum(dim=1).numpy()

    def __del__(self):
        pass

def load_pixel_data(signal_path, bib_path, max_hits=9, batch=256,
                    num_evt=-1, test_frac=0.2, val_frac=0.1, seed=42):
    """Load signal + BIB HDF5 files into merged, split DataLoaders.

    Returns a dict with:
        train, val, test : DataLoaders for the Transformer
        features         : dict {feature_name: (N,) ndarray} for BDT / plots
        labels           : (N,) ndarray
        raw_hit_counts   : (N,) ndarray
        feature_keys     : list of feature names
        idx_train/val/test : index arrays for consistent splits
    """
    from sklearn.model_selection import train_test_split

    sig_ds = PixelClusterDataset(signal_path, num_evt=num_evt,
                                 max_hits=max_hits, label_override=1)
    bib_ds = PixelClusterDataset(bib_path, num_evt=num_evt,
                                 max_hits=max_hits, label_override=0)

    # Cluster-level arrays for BDT / exploration
    sig_feat = sig_ds.cluster_features(CLUSTER_FEATURE_KEYS)
    bib_feat = bib_ds.cluster_features(CLUSTER_FEATURE_KEYS)
    features = {k: np.concatenate([sig_feat[k], bib_feat[k]])
                for k in CLUSTER_FEATURE_KEYS}
    labels = np.concatenate([sig_ds.labels(), bib_ds.labels()])
    raw_counts = np.concatenate([sig_ds.raw_hit_counts(),
                                 bib_ds.raw_hit_counts()])
    print(f"  Loaded {len(labels)} clusters ({sig_ds._n} signal + {bib_ds._n} BIB)")

    # Merge for DataLoader
    combined = torch.utils.data.ConcatDataset([sig_ds, bib_ds])
    n = len(combined)

    # Stratified splits
    indices = np.arange(n)
    idx_tv, idx_test = train_test_split(
        indices, test_size=test_frac, random_state=seed, stratify=labels)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=val_frac / (1 - test_frac),
        random_state=seed, stratify=labels[idx_tv])

    kw = dict(pin_memory=torch.cuda.is_available(), num_workers=0)
    train_loader = DataLoader(
        torch.utils.data.Subset(combined, idx_train),
        batch_size=batch, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(
        torch.utils.data.Subset(combined, idx_val),
        batch_size=batch, shuffle=False, **kw)
    test_loader = DataLoader(
        torch.utils.data.Subset(combined, idx_test),
        batch_size=batch, shuffle=False, **kw)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'features': features,
        'labels': labels,
        'raw_hit_counts': raw_counts,
        'feature_keys': CLUSTER_FEATURE_KEYS,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test,
    }


