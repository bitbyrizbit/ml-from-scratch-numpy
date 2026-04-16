# kept for reference, not used in final implementation

import numpy as np


class BinaryCrossEntropy:
    """
    Binary Cross-Entropy Loss (Log Loss)

    Handles numerical stability via clipping.
    Fully vectorized.
    """

    def __init__(self, eps: float = 1e-15):
        self.eps = eps

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes average BCE loss.

        Parameters:
        y_true : (n_samples, 1)
        y_pred : (n_samples, 1)

        Returns:
        float : loss value
        """

        # Avoid log(0)
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)

        loss = -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

        return loss

    def gradient(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Gradient of BCE wrt weights and bias.

        Uses simplified form for sigmoid + BCE:
        dL/dw = (1/m) * X^T (y_pred - y)
        dL/db = (1/m) * sum(y_pred - y)
        """

        m = X.shape[0]

        error = y_pred - y_true

        dw = (1 / m) * (X.T @ error)
        db = (1 / m) * np.sum(error)

        return dw, db