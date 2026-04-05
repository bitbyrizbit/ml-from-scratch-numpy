"""
Train-only standard scaling to avoid data leakage.
"""

import numpy as np

class StandardScaler:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        # Compute statistics from training data only.
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / (self.std + 1e-8)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
