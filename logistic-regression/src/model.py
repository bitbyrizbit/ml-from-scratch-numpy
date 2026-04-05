"""
Binary logistic regression using batch gradient descent.
Designed for clarity and controlled experimentation.
"""

import numpy as np

from .utils import sigmoid
from .loss import compute_loss
from .optimizer import gradient_descent


class LogisticRegression:
    # Binary Logistic Regression implemented from scratch.
    
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters

        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        
        # Zero initialization is valid here due to convex loss surface.
        # Gradient simplifies to X.T @ (y_hat - y) / n.
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.n_iters):
            linear_output = X @ self.weights + self.bias
            y_pred = sigmoid(linear_output)

            loss = compute_loss(y, y_pred)
            self.loss_history.append(loss)

            self.weights, self.bias = gradient_descent(
                X,
                y,
                self.weights,
                self.bias,
                self.learning_rate
            )

            if epoch % 100 == 0:
                print(f"Epoch {epoch} || Loss: {loss:.6f}")

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probabilities = self.predict_prob(X)
        return (probabilities >= threshold).astype(int)
