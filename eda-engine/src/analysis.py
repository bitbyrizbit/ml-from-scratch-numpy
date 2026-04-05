"""
EDAEngine acts as a thin orchestration layer.
    It does not compute statistics directly - instead it coordinates lower-level modules.
"""

from .statistics import mean, variance, std, min_max
from .correlation import covariance_matrix, correlation_matrix
from .visualization import plot_feature_distributions, plot_correlation_heatmap
import numpy as np

class EDAEngine:
    """
    Central EDA engine that orchestrates statistical
    analysis and visualization.
    """

    def __init__(self, X, feature_names=None):
        if not isinstance(X, np.ndarray):
            raise TypeError("Input data must be a NumPy array")

        if X.ndim != 2:
            raise ValueError("Input array must be 2-dimensional")
        self.X = X
        self.n_samples, self.n_features = X.shape

        self.feature_names = feature_names

    def descriptive_summary(self):
        # Returns a structured numerical summary.
        mu = mean(self.X)
        var = variance(self.X)
        sd = std(self.X)
        min_vals, max_vals = min_max(self.X)
        
        # Iterating manually instead of vector-dumping
        # Output remains readable and structured
        summary = {}
        for i in range(self.X.shape[1]):
            name = self.feature_names[i] if self.feature_names else f"feature_{i}"
            summary[name] = {
                "mean": mu[i],
                "variance": var[i],
                "std": sd[i],
                "min": min_vals[i],
                "max": max_vals[i]
            }
        return summary

    def relational_analysis(self):
        # Computes covariance and correlation matrices.
        return {
            "covariance": covariance_matrix(self.X),
            "correlation": correlation_matrix(self.X)
        }

    def visualize(self):
        # Runs all visualization utilities.
        plot_feature_distributions(self.X, self.feature_names)
        plot_correlation_heatmap(
            correlation_matrix(self.X),
            self.feature_names
        )

