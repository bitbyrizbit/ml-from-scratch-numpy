import numpy as np
from sklearn.datasets import make_classification


def generate_dataset(
    n_samples: int = 500,
    n_features: int = 20,
    random_state: int = 42
):
    """
    Generate a moderately complex binary classification dataset.

    Why this setup:
    - Enough features to observe optimization behavior
    - Not too large → fast experimentation
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(0.75 * n_features),
        n_redundant=int(0.25 * n_features),
        random_state=random_state
    )

    return X, y.reshape(-1, 1)


def standardize(X: np.ndarray) -> np.ndarray:
    """
    Feature scaling improves convergence stability.

    Important for gradient-based methods.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8  # avoid division by zero

    return (X - mean) / std