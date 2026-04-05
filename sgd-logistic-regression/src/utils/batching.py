import numpy as np


def create_batches(X: np.ndarray, y: np.ndarray, batch_size: int):
    """
    Yield mini-batches of data.

    Notes:
    - Assumes data is already shuffled
    - Keeps batching logic separate from optimizers
    """
    n_samples = X.shape[0]

    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        yield X[start:end], y[start:end]