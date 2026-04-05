import numpy as np
from .base import Optimizer
from src.utils.batching import create_batches


class MiniBatchGradientDescent(Optimizer):
    """
    Mini-batch Gradient Descent.

    Characteristics:
    - Balance between SGD and Batch GD
    - Reduced variance vs SGD
    - Faster than full batch in practice
    """

    def __init__(self, lr: float, epochs: int, batch_size: int):
        super().__init__(lr, epochs)
        self.batch_size = batch_size

    def train(self, model, X: np.ndarray, y: np.ndarray):
        self.loss_history = []
        n_samples = X.shape[0]

        for epoch in range(self.epochs):
            # Shuffle at start of every epoch
            indices = np.random.permutation(n_samples)

            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for X_batch, y_batch in create_batches(
                X_shuffled, y_shuffled, self.batch_size
            ):
                y_pred = model.forward(X_batch)
                loss = model.compute_loss(y_batch, y_pred)

                dw, db = model.gradients(X_batch, y_batch, y_pred)

                self._update(model, dw, db)

                self.loss_history.append(loss)

        return self.loss_history