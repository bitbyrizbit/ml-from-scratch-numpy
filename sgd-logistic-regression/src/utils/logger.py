import os
import pandas as pd
from datetime import datetime


def log_experiment(log_path, config: dict, results: dict):
    """
    Append experiment results to a CSV file.

    Overwrites structure only if file doesn't exist.
    Appends rows otherwise.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **config,
        **results
    }

    df = pd.DataFrame([record])

    if os.path.exists(log_path):
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)