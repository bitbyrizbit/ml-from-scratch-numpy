import numpy as np
import random


def set_seed(seed: int = 42):
    """
    Ensures reproducibility across runs.

    Important for fair comparison between optimizers.
    """
    np.random.seed(seed)
    random.seed(seed)