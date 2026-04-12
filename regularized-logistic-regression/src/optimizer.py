import numpy as np
from .losses import BinaryCrossEntropy as BinaryCrossEntropyLoss
from .regularization import Regularization


class GradientDescentOptimizer:
    """
    Handles training loop for logistic regression with optional regularization.
    """

    def __init__(self):
        self.loss_history = []
        
    def train(self, model, X, y):
        """
        Train the model using full-batch gradient descent.
        """
        if model.w is None:
            model._initialize_parameters(X.shape[1])

        loss_fn = BinaryCrossEntropyLoss()
        reg = Regularization(lambda_=model.lambda_, penalty=model.penalty)

        # Reset history (important for multiple runs)
        self.loss_history = []

        y = y.reshape(-1)
        
        for _ in range(model.n_iterations):

            # Forward pass
            y_pred = model.forward(X)

            # Compute base loss
            loss = loss_fn.compute(y, y_pred)

            # Add regularization penalty
            loss += reg.penalty_term(model.w)

            self.loss_history.append(loss)

            # Gradients (data loss)
            m = X.shape[0]
            dw = (1 / m) * (X.T @ (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            # Add regularization gradient
            dw += reg.gradient(model.w)

            # Parameter update
            model.w -= model.learning_rate * dw
            model.b -= model.learning_rate * db

        return model