import numpy as np

class BinaryCrossEntropy:

    def compute(self, y, y_hat):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return -np.mean(
            y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)
        )
    def derivative(self, y, y_hat):
        # kept for reference - not used in training loop
        # model uses simplified dZ2 = A2 - y directly (sigmoid + BCE cancellation)
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return (y_hat - y) / (y_hat * (1 - y_hat))