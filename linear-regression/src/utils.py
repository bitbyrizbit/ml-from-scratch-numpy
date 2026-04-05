import numpy as np

def train_test_split(X, y, test_size=0.2, seed=42):
    # Simple manual train-test split.
    
    np.random.seed(seed)
    m = X.shape[0]
    indices = np.random.permutation(m)

    split = int(m * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
