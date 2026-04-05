import numpy as np

class LinearRegression:
    # Implements multivariate linear regression using batch gradient descent.

    """
    Linear Regression implemented using batch gradient descent.
    
    This implementation does not include:
    - Regularization
    - Mini-batch updates
    - Early stopping
    """

    def __init__(self, n_features: int):
        self.weights: np.ndarray = np.random.randn(n_features) * 0.01
        self.bias: float = 0.0

    def fit(self, X, y, optimizer, learning_rate=0.01, epochs=1000, verbose=True):
        # Trains the model using the provided optimizer.
        losses = optimizer.gradient_descent(
            self,
            X,
            y,
            learning_rate=learning_rate,
            epochs=epochs,
            verbose=verbose
        )
        return losses


    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias
