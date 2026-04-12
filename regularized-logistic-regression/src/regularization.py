import numpy as np


class Regularization:
    """
    Handles L1 and L2 regularization penalties and gradients.

    Supports:
    - None
    - L1 (Lasso)
    - L2 (Ridge)
    """

    def __init__(self, lambda_: float = 0.0, penalty: str = None):
        self.lambda_ = lambda_
        self.penalty = penalty

    def penalty_term(self, w: np.ndarray) -> float:
        """
        Computes regularization penalty value.

        IMPORTANT:
        Bias is NOT included
        """
        w = np.asarray(w)
        if self.penalty is None or self.lambda_ == 0:
            return 0.0

        if self.penalty == "l2":
            return self.lambda_ * np.sum(w ** 2)

        elif self.penalty == "l1":
            return self.lambda_ * np.sum(np.abs(w))

        else:
            raise ValueError("penalty must be one of [None, 'l1', 'l2']")

    def gradient(self, w: np.ndarray) -> np.ndarray:
        """
        Computes gradient of regularization term wrt weights.

        IMPORTANT:
        Returns gradient only for weights (not bias)
        """
        w = np.asarray(w)
        if self.penalty is None or self.lambda_ == 0:
            return np.zeros_like(w)
        if self.penalty == "l2":
            # d/dw (λ ||w||^2) = 2λw
            return 2 * self.lambda_ * w
        elif self.penalty == "l1":
            # Subgradient: sign(w)
            # Define sign(0) = 0
            return self.lambda_ * np.sign(w)
        else:
            raise ValueError("penalty must be one of [None, 'l1', 'l2']")