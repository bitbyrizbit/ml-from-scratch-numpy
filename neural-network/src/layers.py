import numpy as np

class DenseLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))

        self.input = None
        self.Z = None

    def forward(self, X):
        self.input = X
        self.Z = X @ self.W + self.b
        return self.Z

    def backward(self, dZ):
        m = self.input.shape[0]

        dW = (self.input.T @ dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = dZ @ self.W.T

        return dW, db, dX