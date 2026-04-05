import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.
    """
    preds = (y_pred >= 0.5).astype(int)
    return float(np.mean(preds == y_true))


def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Standalone loss (useful for validation comparisons).
    """
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)

    return float(
        -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )
    )