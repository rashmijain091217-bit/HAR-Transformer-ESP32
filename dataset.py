"""
dataset.py — UCI HAR Dataset loader.

Reads the 9-channel inertial-signal .txt files from disk, stacks them
into (N, 128, 9) arrays, and zero-indexes the labels.

NOTE: The UCI HAR dataset is ALREADY windowed into 128-timestep segments.
      No sliding-window is applied here.
"""

import os
import numpy as np
from config import N_CHANNELS, SEQ_LEN, N_CLASSES

# ─────────────────────────────────────────────
# Signal file names (order defines channel axis)
# ─────────────────────────────────────────────
SIGNAL_NAMES = [
    "body_acc_x",  "body_acc_y",  "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def _load_signals(base_path: str, split: str) -> np.ndarray:
    """
    Load all 9 inertial signal files for a given split ('train' or 'test').

    Each file has shape (N_samples, 128); we stack along a new axis → (N, 128, 9).

    Args:
        base_path : Root folder of the UCI HAR dataset.
        split     : 'train' or 'test'.

    Returns:
        np.ndarray of shape (N, SEQ_LEN, N_CHANNELS), dtype float32.
    """
    signals_path = os.path.join(base_path, split, "Inertial Signals")
    channels = []

    for sig in SIGNAL_NAMES:
        fname = f"{sig}_{split}.txt"
        fpath = os.path.join(signals_path, fname)

        # Each row = one window of 128 space-separated float values
        data = np.loadtxt(fpath)           # (N, 128)
        channels.append(data)

    # Stack: list of (N,128) → (9, N, 128) → transpose → (N, 128, 9)
    X = np.stack(channels, axis=0)        # (9, N, 128)
    X = np.transpose(X, (1, 2, 0))        # (N, 128, 9)
    return X.astype(np.float32)


def _load_labels(base_path: str, split: str) -> np.ndarray:
    """
    Load activity labels for a given split.

    UCI HAR labels are 1-indexed (1–6); we subtract 1 for 0-indexed classes.

    Returns:
        np.ndarray of shape (N,), dtype int32, values in [0, 5].
    """
    fname = f"y_{split}.txt"
    fpath = os.path.join(base_path, split, fname)
    labels = np.loadtxt(fpath, dtype=np.int32)
    return labels - 1    # Convert 1-6 → 0-5


def load_dataset(base_path: str = "UCI HAR Dataset"):
    """
    Public API: load train and test sets from the UCI HAR Dataset.

    Args:
        base_path : Path to the extracted 'UCI HAR Dataset' folder.

    Returns:
        X_train : (7352, 128, 9)  float32
        y_train : (7352,)         int32
        X_test  : (2947, 128, 9)  float32
        y_test  : (2947,)         int32
    """
    print(f"[dataset] Loading UCI HAR dataset from: {os.path.abspath(base_path)}")

    X_train = _load_signals(base_path, "train")
    y_train = _load_labels(base_path,  "train")
    X_test  = _load_signals(base_path, "test")
    y_test  = _load_labels(base_path,  "test")

    # ── Sanity checks ──────────────────────────────────────────────────
    assert X_train.shape == (7352, SEQ_LEN, N_CHANNELS), \
        f"Unexpected X_train shape: {X_train.shape}"
    assert X_test.shape  == (2947, SEQ_LEN, N_CHANNELS), \
        f"Unexpected X_test shape: {X_test.shape}"
    assert y_train.min() == 0 and y_train.max() == N_CLASSES - 1
    assert y_test.min()  == 0 and y_test.max()  == N_CLASSES - 1

    print(f"  X_train: {X_train.shape}  |  y_train: {y_train.shape}")
    print(f"  X_test : {X_test.shape}   |  y_test : {y_test.shape}")
    print(f"  Label range: {y_train.min()}–{y_train.max()}  (0-indexed)\n")

    return X_train, y_train, X_test, y_test


def normalize(X_train: np.ndarray, X_test: np.ndarray):
    """
    Channel-wise Z-score normalisation.

    Statistics are computed ONLY on the training split to avoid data leakage.

    Args:
        X_train : (N_train, 128, 9)
        X_test  : (N_test,  128, 9)

    Returns:
        Normalised X_train and X_test, both float32.
    """
    # Compute mean & std across samples AND timesteps per channel
    mean = X_train.mean(axis=(0, 1), keepdims=True)   # (1, 1, 9)
    std  = X_train.std(axis=(0, 1),  keepdims=True)    # (1, 1, 9)
    std  = np.where(std == 0, 1.0, std)                # Avoid division by zero

    X_train_n = (X_train - mean) / std
    X_test_n  = (X_test  - mean) / std

    print(f"[dataset] Normalised — mean≈0, std≈1 per channel (train stats only)")
    return X_train_n.astype(np.float32), X_test_n.astype(np.float32)
