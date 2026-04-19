import numpy as np

from .layers import DenseLayer
from .activations import sigmoid, sigmoid_derivative, relu, relu_derivative
from .losses import BinaryCrossEntropy


class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, learning_rate=0.01, epochs=1000):

        self.lr = learning_rate
        self.epochs = epochs

        self.layer1 = DenseLayer(input_dim, hidden_dim)
        self.layer2 = DenseLayer(hidden_dim, 1)

        self.loss_fn = BinaryCrossEntropy()
        self.loss_history = []

    def forward(self, X):
        Z1 = self.layer1.forward(X)
        A1 = relu(Z1)

        Z2 = self.layer2.forward(A1)
        A2 = sigmoid(Z2)

        return A1, A2

    def backward(self, y, A1, A2):

        # output layer
        dZ2 = A2 - y
        dW2, db2, dA1 = self.layer2.backward(dZ2)

        # hidden layer
        dZ1 = dA1 * relu_derivative(self.layer1.Z)
        dW1, db1, _ = self.layer1.backward(dZ1)

        # update
        self.layer2.W -= self.lr * dW2
        self.layer2.b -= self.lr * db2

        self.layer1.W -= self.lr * dW1
        self.layer1.b -= self.lr * db1

    def fit(self, X, y):

        for _ in range(self.epochs):

            A1, A2 = self.forward(X)

            loss = self.loss_fn.compute(y, A2)
            self.loss_history.append(loss)

            self.backward(y, A1, A2)

        return self

    def predict_proba(self, X):
        _, A2 = self.forward(X)
        return A2

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)