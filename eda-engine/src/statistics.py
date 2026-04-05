"""
Core descriptive statistics implemented manually using NumPy.
    Design decision:
    All operations are vectorized and computed column-wise.
    
    Assumes input is a 2D numeric array of shape (m_samples, n_features).
"""

import numpy as np

def mean(X):
    # Compute column-wise statistical summary.  
    # Implemented explicitly instead of np.mean for clarity.
    return np.sum(X, axis=0) / X.shape[0]


def variance(X):
    # Using population variance (divide by m)
    mu = mean(X)
    return np.sum((X - mu) ** 2, axis=0) / X.shape[0]


def std(X):
    # Standard deviation derived from variance.
    return np.sqrt(variance(X))


def min_max(X):
    # Returns column-wise minimum and maximum.
    return np.min(X, axis=0), np.max(X, axis=0)

def percentile(X, q):
    """
    Computes percentile with linear interpolation.
    q in [0, 100]
    """
    sorted_X = np.sort(X, axis=0)
    m = X.shape[0]

    pos = (q / 100) * (m - 1)
    lower = int(np.floor(pos))
    upper = int(np.ceil(pos))

    if lower == upper:
        return sorted_X[lower]

    weight = pos - lower
    return (1 - weight) * sorted_X[lower] + weight * sorted_X[upper]
