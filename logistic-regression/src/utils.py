"""
Utility functions (numerically stable sigmoid).
"""

import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    # Cliping values prevents overflow in exp.
    
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))
