"""
Binary cross-entropy loss with numerical safeguards.
"""

import numpy as np

def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Parameters:
        y_true : Ground truth labels (0 or 1)
        y_pred : Predicted probabilities

    Returns:
        Scalar loss value
    """
    
    # Small epsilon prevents log(0).
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )

    return loss
