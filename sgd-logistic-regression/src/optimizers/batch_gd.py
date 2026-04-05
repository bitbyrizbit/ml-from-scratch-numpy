import numpy as np
from .base import Optimizer


class BatchGradientDescent(Optimizer):
    """
    Full-batch gradient descent.

    Characteristics:
    - Uses entire dataset per update
    - Low variance updates
    - Smooth but computationally heavier
    """

    def train(self, model, X: np.ndarray, y: np.ndarray):
        self.loss_history = []

        for epoch in range(self.epochs):
            # Forward pass on entire dataset
            y_pred = model.forward(X)

            # Compute loss
            loss = model.compute_loss(y, y_pred)

            # Compute gradients
            dw, db = model.gradients(X, y, y_pred)

            # Update parameters
            self._update(model, dw, db)

            self.loss_history.append(loss)

        return self.loss_history