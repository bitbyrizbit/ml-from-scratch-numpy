"""
Batch gradient descent update logic.
"""

import numpy as np

def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: float,
    learning_rate: float
):
    
    # Performs one step of batch gradient descent for logistic regression.
    # Gradient simplifies to X^T (y_hat - y) for BCE + sigmoid.

    n_samples = X.shape[0]

    y_pred = 1.0 / (1.0 + np.exp(-np.clip(X @ weights + bias, -500, 500)))
    error = y_pred - y

    dw = (1 / n_samples) * np.dot(X.T, error)
    db = (1 / n_samples) * np.sum(error)

    # Standard parameter update.
    weights -= learning_rate * dw
    bias -= learning_rate * db

    return weights, bias
