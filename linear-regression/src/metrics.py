import numpy as np

def mse(y_true, y_pred):
    # Mean Squared Error used as training objective.
      
    """
    Parameters
    y_true : np.ndarray
    y_pred : np.ndarray
    """
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot
