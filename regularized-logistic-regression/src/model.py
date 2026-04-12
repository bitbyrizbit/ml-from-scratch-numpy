import numpy as np


class RegularizedLogisticRegression:
    """
    Logistic Regression model (no training logic here).

    Handles:
    - parameter initialization
    - forward pass
    - prediction
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 2000,
        lambda_: float = 0.0,
        penalty: str = None,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_ = lambda_
        self.penalty = penalty

        self.w = None
        self.b = None

    def _initialize_parameters(self, n_features: int):
        """
        Initialize weights and bias.

        Small random init -> avoids symmetry
        """
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass -> probabilities
        """
        z = X @ self.w + self.b
        return self._sigmoid(z).reshape(-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.forward(X)
        return (probs >= threshold).astype(int)