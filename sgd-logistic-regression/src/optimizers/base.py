from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    """
    Base class for all optimizers.

    Defines a consistent interface so experiments remain clean.
    """

    def __init__(self, lr: float, epochs: int):
        self.lr = lr
        self.epochs = epochs

        # Store training history for analysis
        self.loss_history = []

    @abstractmethod
    def train(self, model, X: np.ndarray, y: np.ndarray):
        """
        Each optimizer must implement its own training strategy.
        """
        pass

    def _update(self, model, dw, db):
        """
        Shared parameter update logic.

        Keeping this here avoids duplication.
        """
        model.w -= self.lr * dw
        model.b -= self.lr * db