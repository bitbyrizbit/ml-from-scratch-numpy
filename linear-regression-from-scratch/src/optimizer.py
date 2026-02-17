import numpy as np

class GradientDescent:
    # Batch Gradient Descent optimizer.
    
    def gradient_descent(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        epochs: int,
        verbose: bool = True,
        log_interval: int = 100
    ) -> list:

        n = len(y)
        losses = []

        for epoch in range(epochs):
            predictions = model.predict(X)
            error = predictions - y

            dw = (1 / n) * np.dot(X.T, error)
            db = (1 / n) * np.sum(error)

            model.weights -= learning_rate * dw
            model.bias -= learning_rate * db

            loss = np.mean(error ** 2)
            losses.append(loss)

            if verbose and epoch % log_interval == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

        return losses
