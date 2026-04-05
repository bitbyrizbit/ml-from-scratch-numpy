import numpy as np
from .base import Optimizer


class StochasticGradientDescent(Optimizer):
    """
    Stochastic Gradient Descent (SGD).

    Characteristics:
    - One sample per update
    - High variance (noisy updates)
    - Faster iterations but unstable trajectory
    """

    def train(self, model, X: np.ndarray, y: np.ndarray):
        self.loss_history = []
        n_samples = X.shape[0]

        for epoch in range(self.epochs):
            # Shuffle every epoch (critical for SGD behavior)
            indices = np.random.permutation(n_samples)

            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_samples):
                xi = X_shuffled[i:i+1]
                yi = y_shuffled[i:i+1]

                y_pred = model.forward(xi)
                loss = model.compute_loss(yi, y_pred)

                dw, db = model.gradients(xi, yi, y_pred)

                self._update(model, dw, db)

                # Track per-update loss (not per epoch)
                self.loss_history.append(loss)

        return self.loss_history