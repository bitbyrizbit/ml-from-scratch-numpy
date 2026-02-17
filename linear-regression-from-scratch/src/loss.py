import numpy as np

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Computes Mean Squared Error between predictions and ground truth.
    
    """
    Parameters:-
    y_true : np.ndarray
        Actual target values.
    y_pred : np.ndarray
        Model predictions.

    Returns:-
    float -> Mean squared error.
    """
    
    return np.mean((y_true - y_pred) ** 2)
