import numpy as np


class LogisticRegression:
    """
    Binary Logistic Regression implemented from scratch.

    Design choice:
    - Model is purely mathematical (no training logic here)
    - Optimizers handle parameter updates externally
    """

    def __init__(self, n_features: int):
        # Initialize parameters to zeros for controlled comparison across optimizers
        self.w = np.zeros((n_features, 1))
        self.b = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid.
        """
        # Avoid overflow for large negative values
        return 1 / (1 + np.exp(-z))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute model predictions (probabilities).
        """
        return self._sigmoid(X @ self.w + self.b)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary cross-entropy loss.

        Clipping avoids log(0) instability.
        """
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)

        loss = -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )
        return float(loss)

    def gradients(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute gradients of loss w.r.t parameters.

        Fully vectorized:
        - No loops over features
        """
        m = X.shape[0]

        error = y_pred - y_true
        dw = (X.T @ error) / m
        db = np.sum(error) / m

        return dw, db

    def parameters(self):
        """
        Convenience method for accessing parameters.
        """
        return self.w, self.b