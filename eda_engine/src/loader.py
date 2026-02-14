"""
A custom CSV loader for numeric datasets.
    Note:
    This loader intentionally avoids pandas to maintain full NumPy control over parsing and data structure.
    
    Assumes clean numeric input.
"""

import numpy as np

def load_wine_data(path: str, return_header: bool = False) -> np.ndarray:
    """
    Loads Wine Quality dataset (comma-separated).
    Pure NumPy implementation.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    # Remove header
    lines = lines[1:]

    # Splits data by comma
    data = [list(map(float, line.strip().split(","))) for line in lines]

    return np.array(data, dtype=float)


def split_features_target(data: np.ndarray, target_index: int = -1):
    # Splits dataset into features (X) and target (y).
    X = data[:, :target_index]
    y = data[:, target_index].reshape(-1, 1)
    return X, y
