# **Experiment Results Summary** 

This document summarizes how different learning rates affect convergence behavior, numerical stability, and generalization performance.

---

## Experiment Configuration

The same dataset, preprocessing pipeline, and number of training iterations were used for all experiments to ensure fair comparison.

- Model: Linear Regression (implemented from scratch)
- Optimization: Batch Gradient Descent
- Loss Function: Mean Squared Error (MSE)
- Feature Scaling: Standardization
- Training Iterations: 1000
- Evaluation Metric: Test Mean Squared Error

---

## Learning Rate = 0.001

With a learning rate of 0.001, the training loss decreases smoothly and monotonically, indicating very stable gradient updates. 
However, convergence is slow, and even after 1000 iterations the model does not fully reach the optimal minimum.

This incomplete convergence leads to a higher test error.
The final test MSE observed was **0.89935**, suggesting mild underfitting.
While numerically safe, this learning rate is inefficient for the given problem and training budget.

---

## Learning Rate = 0.1

A learning rate of 0.1 results in a sharp reduction of training loss within the initial iterations, followed by stable convergence toward the minimum.
The optimizer reaches a near-optimal solution well within the available training iterations.

This setting achieves the best balance between convergence speed and stability.
The final test MSE was **0.3500**, which is significantly lower than that obtained with a smaller learning rate, indicating better generalization to unseen data.

---

## Learning Rate = 1.0

Using a learning rate of 1.0 causes the optimization process to become numerically unstable.
During training, the loss rapidly explodes and reaches extremely large values, triggering runtime warnings related to overflow.

As a result, the model fails to converge and produces no meaningful predictions. 
This behavior demonstrates classic gradient descent divergence caused by excessively large update steps.

---

## Overall Analysis

These experiments highlight the sensitivity of gradient descent to the choice of learning rate.

A very small learning rate ensures stability but risks slow convergence and underfitting. 
An excessively large learning rate leads to divergence and numerical failure. 
A properly tuned learning rate enables both fast convergence and strong generalization.

For this dataset and implementation, a learning rate of **0.1** provides the most effective optimization behavior.

---

## Conclusion

The learning rate is not a cosmetic hyperparameter but a fundamental control mechanism for optimization dynamics. 
Proper selection is essential for achieving reliable and efficient model training.
