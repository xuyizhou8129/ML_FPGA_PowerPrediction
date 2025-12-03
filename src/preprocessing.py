# src/preprocessing.py

from typing import Tuple, List
import numpy as np
import pandas as pd

FEATURE_COLS: List[str] = [
    "hls_synth__latency_best_cycles",
    "hls_synth__latency_average_cycles",
    "hls_synth__latency_worst_cycles",
    "hls_synth__resources_lut_used",
    "hls_synth__resources_ff_used",
    "hls_synth__resources_dsp_used",
    "hls_synth__resources_bram_used",
    "hls_synth__resources_uram_used",
]

TARGET_COL: str = "impl__power__total_power"


def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)

    # Basic sanity check: ensure required columns exist
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy(dtype=float)

    return X, y, FEATURE_COLS.copy()


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    # Percentage of data to use for testing
    random_state: int = 0,
    # Seed for the random number generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert 0.0 < test_size < 1.0

    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_test = int(np.floor(test_size * n_samples))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def standardize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    eps = 1e-12
    std_safe = np.where(std < eps, 1.0, std)

    X_train_std = (X_train - mean) / std_safe
    X_test_std = (X_test - mean) / std_safe

    return X_train_std, X_test_std, mean, std_safe
