"""
Relational statistical measures.
    Implements covariance and Pearson correlation (without relying on np.cov or np.corrcoef).
"""

import numpy as np
from .statistics import mean

def covariance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Computes sample covariance matrix:
    Cov = (1 / (m - 1)) * X_centered^T X_centered
    """
    m = X.shape[0]
    X_centered = X - mean(X)
    return (1 / (m - 1)) * (X_centered.T @ X_centered)


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    # Computes Pearson correlation matrix using covariance.
    
    # Using covariance normalization instead of np.corrcoef to retain explicit control over computation steps.

    cov = covariance_matrix(X)
    std_dev = np.sqrt(np.diag(cov))
    
    # Preventing division by zero in case a feature has zero variance
    denom = np.outer(std_dev, std_dev)
    denom[denom == 0] = 1e-12
    return cov / denom
